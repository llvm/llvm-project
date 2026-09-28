; RUN: split-file %s %t
; RUN: opt -passes=pgo-instr-gen -pgo-instrument-entry -S %t/input.ll | FileCheck %s --check-prefix=GEN
; RUN: %python %t/raw.py > %t/profile.raw
; RUN: llvm-profdata merge %t/profile.raw -o %t/profile
; RUN: opt -passes=pgo-instr-use,verify -pgo-test-profile-file=%t/profile -S %t/input.ll -o %t/default.ll
; RUN: opt -passes=pgo-instr-use,verify -pgo-test-profile-file=%t/profile -pgo-uniformity-metadata=true -S %t/input.ll -o %t/enabled.ll
; RUN: diff %t/default.ll %t/enabled.ll
; RUN: FileCheck %s --check-prefixes=ON,COUNTS < %t/default.ll
; RUN: opt -passes=pgo-instr-use,verify -pgo-test-profile-file=%t/profile -pgo-uniformity-metadata=false -S %t/input.ll | FileCheck %s --check-prefix=COUNTS --implicit-check-not=uniformity.profile
; RUN: opt -passes=pgo-instr-use,verify -pgo-test-profile-file=%t/profile -pgo-uniformity-metadata=false -S %t/default.ll | FileCheck %s --check-prefix=COUNTS --implicit-check-not=uniformity.profile
; RUN: %python %t/raw.py missing > %t/missing.raw
; RUN: llvm-profdata merge %t/missing.raw -o %t/missing.profile
; RUN: opt -passes=pgo-instr-use,verify -pgo-test-profile-file=%t/missing.profile -pgo-uniformity-metadata=false -S %t/default.ll | FileCheck %s --check-prefix=COUNTS --implicit-check-not=uniformity.profile
; RUN: %python %t/raw.py mismatch > %t/mismatch.raw
; RUN: llvm-profdata merge %t/mismatch.raw -o %t/mismatch.profile
; RUN: opt -passes=pgo-instr-use,verify -pgo-test-profile-file=%t/mismatch.profile -pgo-uniformity-metadata=false -S %t/default.ll | FileCheck %s --check-prefix=COUNTS --implicit-check-not=uniformity.profile
; RUN: %python %t/raw.py zero > %t/zero.raw
; RUN: llvm-profdata merge %t/zero.raw -o %t/zero.profile
; RUN: opt -passes=pgo-instr-use,verify -pgo-test-profile-file=%t/zero.profile -pgo-uniformity-metadata=false -S %t/default.ll | FileCheck %s --check-prefix=ZERO --implicit-check-not=uniformity.profile

; The fixture follows the real instrumented counter order. Both block counters
; observe full-wave entries. The select counter is unrelated to block metadata.
; GEN-LABEL: define void @diamond
; GEN: entry:
; GEN: call void @llvm.instrprof.increment({{.*}}i64 [[HASH:942389667449461396]], i32 3, i32 0)
; GEN: b:
; GEN: call void @llvm.instrprof.increment({{.*}}i64 [[HASH]], i32 3, i32 1)
; GEN: call void @llvm.instrprof.increment.step({{.*}}i64 [[HASH]], i32 3, i32 2,

; Disabling uniformity preserves ordinary entry, branch and select counts.
; It also clears existing uniformity annotations before missing, mismatched or zero-count records
; can skip normal annotation. No consumer-specific switch is required.
; ON: define void @diamond({{.*}}!uniformity.profile
; ON: br i1 %cond, {{.*}}!block.uniformity.profile{{.*}}!branch.uniformity.profile
; ON: b:
; ON: br label %exit, {{.*}}!block.uniformity.profile
; COUNTS-DAG: !{!"function_entry_count", i64 6400}
; COUNTS-DAG: !{!"branch_weights", i32 3200, i32 3200}
; COUNTS-DAG: !{!"branch_weights", i32 1600, i32 4800}
; ZERO: !{!"function_entry_count", i64 0}

;--- input.ll
source_filename = "uniformity-profile-use.ll"
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

mode = sys.argv[1] if len(sys.argv) > 1 else "uniform"
name = b"missing" if mode == "missing" else b"diamond"
name_ref = int.from_bytes(hashlib.md5(name).digest()[:8], "little")
# Raw v11, IR entry-first layout; hash checked against pgo-instr-gen above.
func_hash = 942389667449461396
if mode == "mismatch":
    func_hash += 1
lanes = [6400, 3200, 1600]
if mode == "zero":
    lanes = [0, 0, 0]
uniform = lanes[:]
version = 11 | (1 << 56) | (1 << 58)
num_counters = len(lanes)
counter_delta = 72
uniform_delta = counter_delta + num_counters * 8
names_delta = uniform_delta + len(uniform) * 8
names = bytes([len(name), 0]) + name
header = [0xff6c70726f667281, version, 0, 1, 0, num_counters, 0,
          0, 0, len(uniform), 0, uniform_delta, len(names), counter_delta,
          uniform_delta, names_delta, 0, 0, 2]
record = struct.pack("<7QI4HI", name_ref, func_hash, counter_delta,
                     uniform_delta, 0, 0, 0, num_counters, 0, 0, 0, 64, 0)
counts = lanes + uniform
sys.stdout.buffer.write(struct.pack("<19Q", *header) + record +
                        struct.pack("<" + "Q" * len(counts), *counts) +
                        names + bytes((-len(names)) % 8))
