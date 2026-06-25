#!/usr/bin/env python3
"""Pin the default AMDGPU subtarget on bare-triple llc RUN lines.

Some AMDGPU codegen tests run llc with a bare ``-mtriple=amdgcn[-os]`` and no
``-mcpu``, so codegen happens on the triple's *default* subtarget. The default
depends on the OS:

  * amdhsa            -> gfx700
  * everything else   -> gfx600
    (unknown / no-os, mesa3d, amdpal, ...)

The blank default subtarget is a featureless generic that does not match any
explicit ``-mcpu`` exactly, so making it explicit changes codegen output (ELF
e_flags, scheduling, kernel descriptor fields). This script adds the matching
``-mcpu`` so the target is unambiguous; affected expectations must then be
regenerated (e.g. with update_llc_test_checks.py / update_mir_test_checks.py).

This is the first step of a two-step migration: once the subtarget is an
explicit ``-mcpu=gfxN``, the separate triple-folding pass can convert
``-mtriple=amdgcn -mcpu=gfxN`` to the ``amdgpuN.NN`` subarch triple.

A RUN line is rewritten only when it:
  * invokes llc (not opt),
  * has exactly one ``-mtriple=amdgcn`` (bare arch, not a subarch triple),
  * has no ``-mcpu=`` already.

Usage:
    amdgpu-pin-default-subtarget.py [--dry-run] FILE [FILE ...]

Prints each file changed (or that would change, with --dry-run). Exit status
0 if any file changed, 1 otherwise.
"""
import re
import sys

# -mtriple=amdgcn[-vendor[-os...]] with a bare 'amdgcn' arch token.
MTRIPLE_RE = re.compile(r'-mtriple=(amdgcn)(-[\w.-]*)?(?=\s|$)')


def default_mcpu(triple):
    return "gfx700" if "amdhsa" in triple else "gfx600"


def rewrite_run_line(line):
    if not re.search(r'\bllc\b', line):
        return line
    if '-mcpu=' in line:
        return line
    triples = MTRIPLE_RE.findall(line)
    if len(triples) != 1:
        return line
    m = MTRIPLE_RE.search(line)
    full = m.group(0)[len('-mtriple='):]
    cpu = default_mcpu(full)
    # Insert -mcpu right after the -mtriple token.
    return line[:m.end()] + ' -mcpu=' + cpu + line[m.end():]


def process(path, dry_run=False):
    with open(path) as f:
        text = f.read()
    out = []
    changed = False
    for line in text.splitlines(keepends=True):
        if 'RUN:' in line and '-mtriple=amdgcn' in line:
            new = rewrite_run_line(line)
            if new != line:
                changed = True
            out.append(new)
        else:
            out.append(line)
    if changed and not dry_run:
        with open(path, 'w') as f:
            f.write(''.join(out))
    return changed


def main(argv):
    dry_run = False
    if argv and argv[0] == '--dry-run':
        dry_run = True
        argv = argv[1:]
    any_changed = False
    for p in argv:
        if process(p, dry_run):
            print(p)
            any_changed = True
    return 0 if any_changed else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
