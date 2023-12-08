; RUN: opt -passes='print<memoryssa>' -disable-output %s 2>&1 | FileCheck %s

; Note: if @foo is modelled as a MemoryDef, this test will assert with -loop-rotate, due to MemorySSA not
; being preserved when moving instructions that may not read from or write to memory.

; CHECK-LABEL: @main
; CHECK-NOT: MemoryDef
define void @main() {
entry:
  br label %for.cond120

for.cond120:                                      ; preds = %for.body127, %entry
  call void @foo()
  br i1 undef, label %for.body127, label %for.cond.cleanup126

for.cond.cleanup126:                              ; preds = %for.cond120
  unreachable

for.body127:                                      ; preds = %for.cond120
  %0 = load ptr, ptr undef, align 1
  br label %for.cond120
}

declare void @foo() readnone


