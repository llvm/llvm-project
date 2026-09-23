; RUN: split-file %s %t
; RUN: opt -passes=pgo-instr-gen -pgo-instrument-entry -S %t/input.ll | FileCheck %s --check-prefix=GEN
; RUN: %python %t/raw.py wave > %t/wave.raw
; RUN: llvm-profdata merge %t/wave.raw -o %t/wave.profdata
; RUN: opt -passes=pgo-instr-use,verify -pgo-test-profile-file=%t/wave.profdata -S %t/input.ll -o %t/used.ll
; RUN: FileCheck %s --check-prefixes=WAVE,COUNTS < %t/used.ll
; RUN: opt -passes=pgo-instr-use,verify -pgo-test-profile-file=%t/wave.profdata -S %t/used.ll | FileCheck %s --check-prefixes=NONE,COUNTS --implicit-check-not=wave.profile
; RUN: %python %t/raw.py lane > %t/lane.raw
; RUN: llvm-profdata merge %t/lane.raw -o %t/lane.profdata
; RUN: opt -passes=pgo-instr-use,verify -pgo-test-profile-file=%t/lane.profdata -S %t/used.ll | FileCheck %s --check-prefixes=NONE,COUNTS --implicit-check-not=wave.profile
; RUN: %python %t/raw.py bad-wave > %t/bad-wave.raw
; RUN: llvm-profdata merge %t/bad-wave.raw -o %t/bad-wave.profdata
; RUN: opt -passes=pgo-instr-use,verify -pgo-test-profile-file=%t/bad-wave.profdata -S %t/used.ll -o %t/bad-wave.ll 2>&1 | FileCheck %s --check-prefix=WARN
; RUN: FileCheck %s --check-prefixes=NONE,COUNTS --implicit-check-not=wave.profile < %t/bad-wave.ll
; RUN: %python %t/raw.py zero > %t/zero.raw
; RUN: llvm-profdata merge %t/zero.raw -o %t/zero.profdata
; RUN: opt -passes=pgo-instr-use,verify -pgo-test-profile-file=%t/zero.profdata -S %t/input.ll | FileCheck %s --check-prefix=ZERO
; RUN: opt -passes=pgo-instr-use,verify -pgo-test-profile-file=%t/zero.profdata -S %t/used.ll | FileCheck %s --check-prefix=NONE --implicit-check-not=wave.profile
; RUN: %python %t/raw.py no-entry > %t/no-entry.raw
; RUN: llvm-profdata merge %t/no-entry.raw -o %t/no-entry.profdata
; RUN: opt -passes=pgo-instr-use,verify -pgo-test-profile-file=%t/no-entry.profdata -S %t/input.ll | FileCheck %s --check-prefix=NONE --implicit-check-not=wave.profile
; RUN: opt -passes=pgo-instr-use,verify -pgo-test-profile-file=%t/no-entry.profdata -S %t/used.ll | FileCheck %s --check-prefix=NONE --implicit-check-not=wave.profile
; RUN: %python %t/raw.py missing > %t/missing.raw
; RUN: llvm-profdata merge %t/missing.raw -o %t/missing.profdata
; RUN: opt -passes=pgo-instr-use,verify -pgo-test-profile-file=%t/missing.profdata -S %t/used.ll | FileCheck %s --check-prefix=NONE --implicit-check-not=wave.profile
; RUN: %python %t/raw.py hash > %t/hash.raw
; RUN: llvm-profdata merge %t/hash.raw -o %t/hash.profdata
; RUN: opt -passes=pgo-instr-use,verify -pgo-test-profile-file=%t/hash.profdata -S %t/used.ll -o %t/hash.ll 2>&1 | FileCheck %s --check-prefix=MISMATCH
; RUN: FileCheck %s --check-prefix=NONE --implicit-check-not=wave.profile < %t/hash.ll
; RUN: %python %t/raw.py bad-lane > %t/bad-lane.raw
; RUN: llvm-profdata merge %t/bad-lane.raw -o %t/bad-lane.profdata
; RUN: opt -passes=pgo-instr-use,verify -pgo-test-profile-file=%t/bad-lane.profdata -S %t/used.ll -o %t/bad-lane.ll 2>&1 | FileCheck %s --check-prefix=BAD-LANE
; RUN: FileCheck %s --check-prefix=NONE --implicit-check-not=wave.profile < %t/bad-lane.ll
; RUN: opt -mtriple=x86_64-unknown-linux-gnu -passes=pgo-instr-use,verify -pgo-test-profile-file=%t/wave.profdata -S %t/used.ll | FileCheck %s --check-prefix=NONE --implicit-check-not=wave.profile
; RUN: opt -passes=pgo-instr-gen -pgo-instrument-entry -S %t/critical.ll | FileCheck %s --check-prefix=CGEN
; RUN: %python %t/raw.py critical > %t/critical.raw
; RUN: llvm-profdata merge %t/critical.raw -o %t/critical.profdata
; RUN: opt -passes=pgo-instr-use,verify -pgo-test-profile-file=%t/critical.profdata -S %t/critical.ll | FileCheck %s --check-prefix=CRIT

; RUN: %python %t/raw.py skew > %t/skew.raw
; RUN: llvm-profdata merge %t/skew.raw -o %t/skew.profdata
; RUN: opt -passes=pgo-instr-use,verify -pgo-test-profile-file=%t/skew.profdata -S %t/input.ll -o %t/skew.ll
; RUN: FileCheck %s --check-prefix=SKEW < %t/skew.ll
; RUN: opt -passes=pgo-instr-use,verify -pgo-test-profile-file=%t/skew.profdata -S %t/skew.ll -o %t/reapplied.ll
; RUN: FileCheck %s --check-prefix=NONE --implicit-check-not=wave.profile < %t/reapplied.ll
; RUN: opt -passes=pgo-instr-use,verify -pgo-test-profile-file=%t/skew.profdata -S %t/reapplied.ll | FileCheck %s --check-prefix=NONE --implicit-check-not=wave.profile
; RUN: opt -passes=pgo-instr-use,verify -pgo-test-profile-file=%t/wave.profdata -S %t/skew.ll | FileCheck %s --check-prefix=NONE --implicit-check-not=wave.profile
; RUN: %python %t/raw.py skew-lane > %t/skew-lane.raw
; RUN: llvm-profdata merge %t/skew-lane.raw -o %t/skew-lane.profdata
; RUN: opt -passes=pgo-instr-use,verify -pgo-test-profile-file=%t/skew-lane.profdata -S %t/input.ll -o %t/skew-lane.ll
; RUN: opt -passes=pgo-instr-use,verify -pgo-test-profile-file=%t/skew.profdata -S %t/skew-lane.ll | FileCheck %s --check-prefix=NONE --implicit-check-not=wave.profile

; Wave slots follow instrumentation indices, not IR block order. Only entry and
; b have block counters. The trailing select counter must not make exit measured.
; Generation checks keep the raw fixture aligned with the actual producer.
; GEN-LABEL: define void @diamond
; GEN: entry:
; GEN: call void @llvm.instrprof.increment({{.*}}i64 [[HASH:942389667449461396]], i32 3, i32 0)
; GEN: b:
; GEN: call void @llvm.instrprof.increment({{.*}}i64 [[HASH]], i32 3, i32 1)
; GEN: exit:
; GEN: call void @llvm.instrprof.increment.step({{.*}}i64 [[HASH]], i32 3, i32 2,

; A measured zero is retained even when all lane counters are zero. Blocks with
; no measurements remain unavailable. Replacing a profile clears stale metadata
; for absent, incompatible, lane-only, and unsupported-target records.
; NONE: define void @diamond
; WAVE-LABEL: define void @diamond
; WAVE-SAME: !wave.profile ![[WAVES:[0-9]+]]
; WAVE: br i1 %cond, label %a, label %b, !prof ![[WEIGHTS:[0-9]+]]{{.*}}!wave.profile.block ![[ENTRY:[0-9]+]]
; WAVE: br label %exit, !wave.profile.block ![[A:[0-9]+]]
; WAVE: br label %exit, {{.*}}!wave.profile.block ![[B:[0-9]+]]
; WAVE: ret void, !wave.profile.block ![[EXIT:[0-9]+]]
; WAVE-DAG: ![[WAVES]] = distinct !{i64 2, i64 [[FID:-?[0-9]+]], i64 100, i64 0, i64 100, i64 0}
; WAVE-DAG: ![[ENTRY]] = !{i64 2, i64 [[FID]], i64 0, i64 1, i64 1, i64 2}
; WAVE-DAG: ![[A]] = !{i64 2, i64 [[FID]], i64 1, i64 0, i64 3}
; WAVE-DAG: ![[B]] = !{i64 2, i64 [[FID]], i64 2, i64 1, i64 3}
; WAVE-DAG: ![[EXIT]] = !{i64 2, i64 [[FID]], i64 3, i64 0}
; COUNTS-DAG: !{!"branch_weights", i32 3200, i32 3200}
; COUNTS-DAG: !{!"branch_weights", i32 1600, i32 4800}
; ZERO-LABEL: define void @diamond
; ZERO-SAME: !wave.profile ![[WAVES:[0-9]+]]
; ZERO: br i1 %cond, label %a, label %b, {{.*}}!wave.profile.block ![[ENTRY:[0-9]+]]
; ZERO: br label %exit, !wave.profile.block ![[A:[0-9]+]]
; ZERO: br label %exit, {{.*}}!wave.profile.block ![[B:[0-9]+]]
; ZERO: ret void, !wave.profile.block ![[EXIT:[0-9]+]]
; ZERO-DAG: ![[WAVES]] = distinct !{i64 2, i64 [[FID:-?[0-9]+]], i64 0, i64 0, i64 0, i64 0}
; ZERO-DAG: ![[ENTRY]] = !{i64 2, i64 [[FID]], i64 0, i64 1, i64 1, i64 2}
; ZERO-DAG: ![[A]] = !{i64 2, i64 [[FID]], i64 1, i64 0, i64 3}
; ZERO-DAG: ![[B]] = !{i64 2, i64 [[FID]], i64 2, i64 1, i64 3}
; ZERO-DAG: ![[EXIT]] = !{i64 2, i64 [[FID]], i64 3, i64 0}
; WARN: Inconsistent number of wave counts in diamond; ignoring wave profile
; MISMATCH: function control flow change detected (hash mismatch) diamond
; BAD-LANE: Inconsistent number of counts in diamond

; Existing branch weights can change the instrumentation MST without changing
; the CFG hash. Reapplying this skewed profile would move counter 1 from b to a.
; Drop wave metadata on previously profiled functions, including after a lane-only
; load, and do not recreate it on a subsequent load.
; SKEW-LABEL: define void @diamond
; SKEW-SAME: !wave.profile ![[WAVES:[0-9]+]]
; SKEW: br i1 %cond, label %a, label %b, !prof ![[WEIGHTS:[0-9]+]]
; SKEW: a:
; SKEW: br label %exit, !wave.profile.block ![[A:[0-9]+]]
; SKEW: b:
; SKEW: br label %exit, {{.*}}!wave.profile.block ![[B:[0-9]+]]
; SKEW-DAG: ![[WAVES]] = distinct !{i64 2, i64 [[FID:-?[0-9]+]], i64 100, i64 0, i64 90, i64 0}
; SKEW-DAG: ![[WEIGHTS]] = !{!"branch_weights", i32 640, i32 5760}
; SKEW-DAG: ![[A]] = !{i64 2, i64 [[FID]], i64 1, i64 0, i64 3}
; SKEW-DAG: ![[B]] = !{i64 2, i64 [[FID]], i64 2, i64 1, i64 3}

; Critical edges are split before mapping counter indices to block identities.
; CGEN-LABEL: define void @critical
; CGEN: entry:
; CGEN: call void @llvm.instrprof.increment({{.*}}i64 [[CHASH:784007059655560962]], i32 2, i32 0)
; CGEN: entry.exit_crit_edge:
; CGEN: call void @llvm.instrprof.increment({{.*}}i64 [[CHASH]], i32 2, i32 1)
; CRIT-LABEL: define void @critical
; CRIT-SAME: !wave.profile ![[CWAVES:[0-9]+]]
; CRIT: br i1 %cond, label %a, label %entry.exit_crit_edge, {{.*}}!wave.profile.block ![[CENT:[0-9]+]]
; CRIT: entry.exit_crit_edge:
; CRIT: br label %exit, {{.*}}!wave.profile.block ![[CEDGE:[0-9]+]]
; CRIT: a:
; CRIT: br label %exit, !wave.profile.block ![[CA:[0-9]+]]
; CRIT: ret void, !wave.profile.block ![[CEXIT:[0-9]+]]
; CRIT-DAG: ![[CWAVES]] = distinct !{i64 2, i64 [[CFID:-?[0-9]+]], i64 100, i64 25, i64 0, i64 0}
; CRIT-DAG: ![[CENT]] = !{i64 2, i64 [[CFID]], i64 0, i64 1, i64 2, i64 1}
; CRIT-DAG: ![[CEDGE]] = !{i64 2, i64 [[CFID]], i64 1, i64 1, i64 3}
; CRIT-DAG: ![[CA]] = !{i64 2, i64 [[CFID]], i64 2, i64 0, i64 3}
; CRIT-DAG: ![[CEXIT]] = !{i64 2, i64 [[CFID]], i64 3, i64 0}

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

;--- critical.ll
source_filename = "wave-profile-critical.ll"
target triple = "amdgcn-amd-amdhsa"
define void @critical(i1 %cond, ptr %p) {
entry:
  br i1 %cond, label %a, label %exit, !prof !0
a:
  store volatile i32 1, ptr %p
  br label %exit
exit:
  ret void
}

!0 = !{!"branch_weights", i32 1000000, i32 1}

;--- raw.py
import hashlib
import struct
import sys

mode = sys.argv[1]
name = {"missing": b"missing", "critical": b"critical"}.get(mode, b"diamond")
name_ref = int.from_bytes(hashlib.md5(name).digest()[:8], "little")
# Hash emitted by pgo-instr-gen for input.ll; raw v12 has an 80-byte record.
func_hash = 942389667449461396
if mode == "hash":
    func_hash += 1
lanes = [6400, 3200, 1600]
waves = [100, 100, 100]
if mode == "critical":
    func_hash = 784007059655560962
    lanes = [6400, 800]
    waves = [100, 25]
if mode in ("skew", "skew-lane"):
    lanes = [6400, 5760, 1600]
    waves = [100, 90, 100]
if mode in ("lane", "skew-lane"):
    waves = []
elif mode == "bad-wave":
    waves.pop()
elif mode == "bad-lane":
    lanes.pop()
elif mode == "zero":
    lanes = [0, 0, 0]
    waves = [0, 0, 0]
version = 12 | (1 << 56)
if mode != "no-entry":
    version |= 1 << 58
num_counters = len(lanes) + len(waves)
counter_delta = 80
uniform_delta = counter_delta + num_counters * 8
names_delta = uniform_delta + len(lanes) * 8
names = bytes([len(name), 0]) + name
header = [0xff6c70726f667281, version, 0, 1, 0, num_counters, 0,
          0, 0, len(lanes), 0, uniform_delta, len(names), counter_delta,
          uniform_delta, names_delta, 0, 0, 2]
record = struct.pack("<7QI4HII4x", name_ref, func_hash, counter_delta,
                     uniform_delta, 0, 0, 0, num_counters, 0, 0, 0, 64, 0,
                     len(waves))
counts = lanes + waves + [0] * len(lanes)
sys.stdout.buffer.write(struct.pack("<19Q", *header) + record +
                        struct.pack("<" + "Q" * len(counts), *counts) +
                        names + bytes((-len(names)) % 8))
