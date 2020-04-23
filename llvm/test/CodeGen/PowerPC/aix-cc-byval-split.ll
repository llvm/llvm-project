; RUN: not --crash llc -mtriple powerpc-ibm-aix-xcoff -stop-after=machine-cp \
; RUN:   -mcpu=pwr4 -mattr=-altivec -verify-machineinstrs 2>&1 < %s | FileCheck  %s

; RUN: not --crash llc -mtriple powerpc64-ibm-aix-xcoff -stop-after=machine-cp \
; RUN:   -mcpu=pwr4 -mattr=-altivec -verify-machineinstrs 2>&1 < %s | FileCheck  %s

; CHECK: LLVM ERROR: Passing ByVals split between registers and stack not yet implemented.

%struct.Spill = type { [12 x i64 ] }
@GS = external global %struct.Spill, align 4

define i64 @test(%struct.Spill* byval(%struct.Spill) align 4 %s) {
entry:
  %arrayidx_a = getelementptr inbounds %struct.Spill, %struct.Spill* %s, i32 0, i32 0, i32 2
  %arrayidx_b = getelementptr inbounds %struct.Spill, %struct.Spill* %s, i32 0, i32 0, i32 10
  %a = load i64, i64* %arrayidx_a
  %b = load i64, i64* %arrayidx_b
  %add = add i64 %a, %b
  ret i64 %add
}
