; RUN: split-file %s %t
; RUN: %python %t/raw.py dense > %t/dense.raw
; RUN: llvm-profdata merge %t/dense.raw -o %t/dense.profdata
; RUN: opt -passes=pgo-instr-use,verify -pgo-test-profile-file=%t/dense.profdata -S %t/input.ll -o %t/dense.ll
; RUN: FileCheck %s --check-prefix=COUNTS < %t/dense.ll
; RUN: FileCheck %s --check-prefix=UNIFORM < %t/dense.ll
; RUN: opt -passes=pgo-instr-use,verify -pgo-test-profile-file=%t/dense.profdata -pgo-instrument-dense-wave-counts=false -S %t/input.ll -o %t/disabled-gen.ll
; RUN: diff %t/dense.ll %t/disabled-gen.ll
; RUN: %python %t/raw.py no-entry > %t/no-entry.raw
; RUN: llvm-profdata merge %t/no-entry.raw -o %t/no-entry.profdata
; RUN: opt -passes=pgo-instr-use,verify -pgo-test-profile-file=%t/no-entry.profdata -S %t/input.ll | FileCheck %s --check-prefix=COUNTS
; RUN: %python %t/raw.py sparse > %t/sparse.raw
; RUN: not llvm-profdata merge %t/dense.raw %t/sparse.raw -o %t/mixed 2>&1 | FileCheck %s --check-prefix=MIXED
; RUN: not llvm-profdata merge %t/sparse.raw %t/dense.raw -o %t/mixed 2>&1 | FileCheck %s --check-prefix=MIXED
; RUN: cat %t/dense.raw %t/sparse.raw > %t/mixed.raw
; RUN: not llvm-profdata merge %t/mixed.raw -o %t/mixed 2>&1 | FileCheck %s --check-prefix=MIXED
; RUN: cat %t/sparse.raw %t/dense.raw > %t/mixed.raw
; RUN: not llvm-profdata merge %t/mixed.raw -o %t/mixed 2>&1 | FileCheck %s --check-prefix=MIXED

; The profile flags select the layout independently of generation options.
; Appended slots do not change scalar/select counts or uniformity coverage.
; MIXED: cannot merge dense and sparse wave instrumentation layouts
; COUNTS-DAG: !{!"function_entry_count", i64 6400}
; COUNTS-DAG: !{!"branch_weights", i32 3200, i32 3200}
; COUNTS-DAG: !{!"branch_weights", i32 1600, i32 4800}
; UNIFORM: define void @diamond({{.*}}!uniformity.profile
; UNIFORM: br i1 %cond, {{.*}}!block.uniformity.profile{{.*}}!branch.uniformity.profile
; UNIFORM: b:
; UNIFORM: br label %exit, {{.*}}!block.uniformity.profile

;--- input.ll
source_filename = "wave-profile-use.ll"
target triple = "amdgcn-amd-amdhsa"
define void @diamond(i1 %cond, i1 %select_cond, ptr %p) {
entry:
  br i1 %cond, label %a, label %b
a:
  store volatile i32 1, ptr %p
  br label %exit
b:
  store volatile i32 2, ptr %p
  br label %exit
exit:
  %value = select i1 %select_cond, i32 1, i32 2
  store volatile i32 %value, ptr %p
  ret void
}

;--- raw.py
import hashlib
import struct
import sys

mode = sys.argv[1]
name = b"diamond"
name_ref = int.from_bytes(hashlib.md5(name).digest()[:8], "little")
# Preserve the sparse block/select prefix and append two zero lane steps.
lanes = [6400, 3200, 1600]
waves = [100, 50, 100]
version = 12 | (1 << 56) | (1 << 58)
if mode != "sparse":
    version |= 1 << 54
    lanes += [0, 0]
    waves += [100, 100]
if mode == "no-entry":
    version &= ~(1 << 58)
    lanes[0] = 3200
num_counters = len(lanes) + len(waves)
counter_delta = 80
uniform_delta = counter_delta + num_counters * 8
names_delta = uniform_delta + len(lanes) * 8
names = bytes([len(name), 0]) + name
header = [0xff6c70726f667281, version, 0, 1, 0, num_counters, 0,
          0, 0, len(lanes), 0, uniform_delta, len(names), counter_delta,
          uniform_delta, names_delta, 0, 0, 2]
record = struct.pack("<7QI4HII4x", name_ref, 942389667449461396, counter_delta,
                     uniform_delta, 0, 0, 0, num_counters, 0, 0, 0, 64, 0,
                     len(waves))
counts = lanes + waves + lanes
sys.stdout.buffer.write(struct.pack("<19Q", *header) + record +
                        struct.pack("<" + "Q" * len(counts), *counts) +
                        names + bytes((-len(names)) % 8))
