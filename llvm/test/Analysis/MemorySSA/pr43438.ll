; RUN: opt -disable-output -licm -print-memoryssa -enable-mssa-loop-dependency=true < %s 2>&1 | FileCheck %s
target triple = "x86_64-unknown-linux-gnu"

; CHECK-LABEL: @main()
; CHECK: 5 = MemoryPhi(
; CHECK-NOT: 7 = MemoryPhi(
@v_67 = external dso_local global i32, align 1
@v_76 = external dso_local global i16, align 1
@v_86 = external dso_local global i16 *, align 1

define dso_local void @main() {
entry:
  %v_59 = alloca i16, align 2
  br label %for.cond

for.cond:                                         ; preds = %for.body, %entry
  br i1 undef, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  store i16 undef, i16* %v_59, align 2
  br label %for.cond

for.end:                                          ; preds = %for.cond
  br i1 undef, label %if.else568, label %cond.end82

cond.false69:                                     ; No predecessors!
  br label %cond.end82

cond.end82:                                       ; preds = %cond.false69, %cond.true55
  br i1 undef, label %if.else568, label %land.lhs.true87

land.lhs.true87:                                  ; preds = %cond.end82
  br i1 undef, label %if.then88, label %if.else568

if.then88:                                        ; preds = %land.lhs.true87
  store i16 * @v_76, i16 ** @v_86, align 1
  br label %if.end569

if.else568:                                       ; preds = %land.lhs.true87, %cond.end82, %for.end
  store volatile i32 undef, i32 * @v_67, align 1
  br label %if.end569

if.end569:                                        ; preds = %if.else568, %if.then88
  ret void
}

