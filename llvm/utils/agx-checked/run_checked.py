#!/usr/bin/env python3
"""Runs kernels built with the AGX backend's agx-checked feature on the AGX Performance Simulator's agx_run, doing
what a runtime does for checked mode (llvm/docs/AGXUsage.rst, "Checked mode"): it fills the hidden arguments (the
error record, the threadgroup memory bytes and the bounds of each slot, from the buffers it maps), runs the launch
and decodes the error record.

As a library: launch(obj, kernel, args, grid, wg, ...) -> Result. As a script, the scenario suite in
checked_tests.py uses it.
"""

import os
import struct
import subprocess
import tempfile

KINDS = {1: "out of bounds", 2: "null", 3: "misaligned", 4: "threadgroup out of bounds", 5: "stack out of bounds"}
RECORD_VA = 0x7000000
GLOBALS_VA = 0x8000000
BUFFER_VA = 0x1000000  # buffers at 0x1000000 * (k + 1)


class Kernel:
    def __init__(self):
        self.name = ""
        self.args = []  # (kind, bytes, half)
        self.tg_half = 0
        self.glob_half = 0xFFFFFFFF
        self.tg_bytes = 0
        self.checked = None  # (record half, tg bytes half, [(arg, offset, lo half, hi half)])


def kernels(obj_path, objcopy):
    with tempfile.TemporaryDirectory() as t:
        sec = os.path.join(t, "k.bin")
        subprocess.run([objcopy, "--dump-section", ".agx.kernels=" + sec, obj_path, os.path.join(t, "u.o")],
                       check=True)
        d = open(sec, "rb").read()
    out, pos = {}, 0
    while pos + 8 <= len(d):
        size, _version = struct.unpack_from("<II", d, pos)
        if size == 0:
            break
        end = pos + size
        k = Kernel()
        (nlen,) = struct.unpack_from("<I", d, pos + 8)
        k.name = d[pos + 12:pos + 12 + nlen].decode()
        q = pos + 12 + (nlen + 3) // 4 * 4
        _halves, k.tg_half, k.glob_half, _gprs, k.tg_bytes, _stack, nargs = struct.unpack_from("<7I", d, q)
        q += 28
        k.args = [struct.unpack_from("<3I", d, q + 12 * i) for i in range(nargs)]
        q += 12 * nargs
        while q < end:
            (tag,) = struct.unpack_from("<I", d, q)
            if tag == 1:
                q += 16
            elif tag == 2:
                _v, n, rec, tgb = struct.unpack_from("<4I", d, q + 4)
                slots = [struct.unpack_from("<4I", d, q + 20 + 16 * i) for i in range(n)]
                k.checked = (rec, tgb, slots)
                q += 20 + 16 * n
            else:
                break
        out[k.name] = k
        pos = end
    return out


def section(obj_path, name, objcopy):
    with tempfile.TemporaryDirectory() as t:
        out = os.path.join(t, "s.bin")
        r = subprocess.run([objcopy, "-O", "binary", "--only-section=" + name, obj_path, out], capture_output=True)
        return open(out, "rb").read() if r.returncode == 0 and os.path.exists(out) else b""


def symbol(obj_path, name, readelf):
    syms = subprocess.run([readelf, "-sW", obj_path], capture_output=True, text=True, check=True).stdout
    for line in syms.splitlines():
        f = line.split()
        if len(f) >= 8 and f[7] == name:
            return int(f[1], 16), int(f[2])
    return None


class Result:
    def __init__(self, record, buffers, sites):
        claimed, kind, addr, size, site = struct.unpack_from("<IIQII", record, 0)
        self.claimed = claimed
        self.kind = kind & 0xFF
        self.store = bool(kind & 0x100)
        self.atomic = bool(kind & 0x200)
        self.addr = addr
        self.size = size
        self.site = site
        self.thread = struct.unpack_from("<3I", record, 24)
        self.threadgroup = struct.unpack_from("<3I", record, 36)
        self.buffers = buffers
        self.sites = sites

    def site_text(self):
        for line in self.sites.splitlines():
            f = line.split("\t")
            if f and f[0] == str(self.site):
                return "\t".join(f[1:])
        return ""

    def __str__(self):
        if not self.claimed:
            return "no error"
        what = "atomic" if self.atomic else "store" if self.store else "load"
        return "%s %s of %d bytes at %#x, thread %s threadgroup %s, site %d (%s)" % (
            KINDS.get(self.kind, "kind %d" % self.kind), what, self.size, self.addr, self.thread, self.threadgroup,
            self.site, self.site_text())


def launch(obj_path, kernel, args, grid, wg, agx_run, objcopy, readelf, tg_bytes=None, claimed=0, strict=None,
           dynamic_shared=0, pre_buffers=()):
    """args: per source argument, ('buf', bytes) | ('ptr', value) | ('i32', v) | ('i64', v) |
    ('param', [field, ...]) where a field is ('buf', bytes) or ('i32', v) packed in order (by-reference args).
    pre_buffers are mapped first, at BUFFER_VA, BUFFER_VA * 2 ..., for 'ptr' arguments into them.
    Returns a Result with the error record and every buffer's final contents."""
    k = kernels(obj_path, objcopy)[kernel]
    with tempfile.TemporaryDirectory() as t:
        text = os.path.join(t, "text.bin")
        subprocess.run([objcopy, "-O", "binary", "--only-section=.text", obj_path, text], check=True)
        start, _ = symbol(obj_path, kernel, readelf)
        code = os.path.join(t, "code.bin")
        open(code, "wb").write(open(text, "rb").read()[start:])
        cmd = [agx_run, code, "--grid", "%d,%d,%d" % grid, "--wg", "%d,%d,%d" % wg]
        if strict:
            cmd += ["--agx-strict", strict]
        uniforms = {}
        allocs = []  # (lo, hi)
        dumps = []

        def buffer(data):
            va = BUFFER_VA * (len(allocs) + 1)
            f = os.path.join(t, "b%d.bin" % len(allocs))
            open(f, "wb").write(data)
            cmd.extend(["--map", "%#x=%s" % (va, f), "--dump", "%#x=%d:%s.out" % (va, len(data), f)])
            dumps.append(f + ".out")
            allocs.append((va, va + len(data)))
            return va

        def put(half, value, words):
            for w in range(words):
                uniforms[half // 2 + w] = (value >> (32 * w)) & 0xFFFFFFFF

        for data in pre_buffers:
            buffer(data)
        arg_bytes = []
        for (kind, *v), (akind, nbytes, half) in zip(args, k.args):
            if kind == "param":
                blob = b""
                for fk, fv in v[0]:
                    blob += struct.pack("<Q", buffer(fv)) if fk == "buf" else struct.pack("<Q" if fk == "ptr" else "<I", fv)
                blob = blob.ljust(nbytes, b"\0")
                arg_bytes.append(blob)
                if akind == 1:
                    raise RuntimeError("by-pointer arguments are not supported by this harness")
                for w in range(0, nbytes, 4):
                    uniforms[half // 2 + w // 4] = struct.unpack_from("<I", blob, w)[0]
                continue
            val = buffer(v[0]) if kind == "buf" else v[0]
            width = 8 if kind in ("buf", "ptr", "i64") else 4
            arg_bytes.append(struct.pack("<Q" if width == 8 else "<I", val & (2 ** (8 * width) - 1)))
            put(half, val, width // 4)
        for d in range(3):
            uniforms[k.tg_half // 2 + d] = -(-grid[d] // wg[d])
        globals_data = section(obj_path, ".agx.globals", objcopy)
        if k.glob_half != 0xFFFFFFFF:
            g = os.path.join(t, "globals.bin")
            open(g, "wb").write(globals_data.ljust(16, b"\0"))
            cmd += ["--map", "%#x=%s" % (GLOBALS_VA, g)]
            put(k.glob_half, GLOBALS_VA, 2)
        if k.checked:
            rec_half, tgb_half, slots = k.checked
            rec = os.path.join(t, "record.bin")
            open(rec, "wb").write(struct.pack("<I", claimed).ljust(64, b"\0"))
            cmd += ["--map", "%#x=%s" % (RECORD_VA, rec), "--dump", "%#x=64:%s.out" % (RECORD_VA, rec)]
            put(rec_half, RECORD_VA, 2)
            put(tgb_half, (k.tg_bytes if tg_bytes is None else tg_bytes) + dynamic_shared, 1)
            for arg, off, lo_half, hi_half in slots:
                b = arg_bytes[arg]
                p = struct.unpack_from("<Q", b, 0 if off == 0xFFFFFFFF else off)[0]
                lo = hi = p
                for alo, ahi in allocs:
                    if alo <= p < ahi:
                        lo, hi = alo, ahi
                put(lo_half, lo, 2)
                put(hi_half, hi, 2)
        for u, v in sorted(uniforms.items()):
            cmd += ["--uniform", "%d=%d" % (u, v)]
        shared = (k.tg_bytes if tg_bytes is None else tg_bytes) + dynamic_shared
        if shared:
            cmd += ["--shared", str(shared)]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if r.returncode:
            raise RuntimeError("agx_run failed: " + (r.stdout + r.stderr)[-600:])
        record = open(rec + ".out", "rb").read() if k.checked else bytes(64)
        bufs = [open(o, "rb").read() for o in dumps]
        sites_at = symbol(obj_path, "__agx_checked_sites", readelf)
        sites = ""
        if sites_at:
            off, size = sites_at
            sites = globals_data[off:off + size].split(b"\0")[0].decode()
        return Result(record, bufs, sites)
