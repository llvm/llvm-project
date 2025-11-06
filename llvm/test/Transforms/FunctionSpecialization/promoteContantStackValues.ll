; RUN: opt -passes="ipsccp<func-spec>,inline,instcombine" -force-specialization -funcspec-max-iters=1 -S < %s | FileCheck %s --check-prefix=ITERS1
; RUN: opt -passes="ipsccp<func-spec>,inline,instcombine" -force-specialization -funcspec-max-iters=2 -S < %s | FileCheck %s --check-prefix=ITERS2
; RUN: opt -passes="ipsccp<func-spec>,inline,instcombine" -force-specialization -funcspec-max-iters=3 -S < %s | FileCheck %s --check-prefix=ITERS3
; RUN: opt -passes="ipsccp<func-spec>,inline,instcombine" -force-specialization -funcspec-max-iters=4 -S < %s | FileCheck %s --check-prefix=ITERS4

@low = internal constant i32 0, align 4
@high = internal constant i32 6, align 4

define internal void @recursiveFunc(ptr nocapture readonly %lo, i32 %step, ptr nocapture readonly %hi) {
  %lo.temp = alloca i32, align 4
  %hi.temp = alloca i32, align 4
  %lo.load = load i32, ptr %lo, align 4
  %hi.load = load i32, ptr %hi, align 4
  %cmp = icmp ne i32 %lo.load, %hi.load
  br i1 %cmp, label %block6, label %ret.block

block6:
  call void @print_val(i32 %lo.load, i32 %hi.load)
  %add = add nsw i32 %lo.load, %step
  %sub = sub nsw i32 %hi.load, %step
  store i32 %add, ptr %lo.temp, align 4
  store i32 %sub, ptr %hi.temp, align 4
  call void @recursiveFunc(ptr nonnull %lo.temp, i32 %step, ptr nonnull %hi.temp)
  br label %ret.block

ret.block:
  ret void
}

; ITERS1:  @specialized.arg.1 = internal constant i32 0
; ITERS1:  @specialized.arg.2 = internal constant i32 6
; ITERS1:  @specialized.arg.3 = internal constant i32 1
; ITERS1:  @specialized.arg.4 = internal constant i32 5

; ITERS2:  @specialized.arg.1 = internal constant i32 0
; ITERS2:  @specialized.arg.2 = internal constant i32 6
; ITERS2:  @specialized.arg.3 = internal constant i32 1
; ITERS2:  @specialized.arg.4 = internal constant i32 5
; ITERS2:  @specialized.arg.5 = internal constant i32 2
; ITERS2:  @specialized.arg.6 = internal constant i32 4

; ITERS3:  @specialized.arg.1 = internal constant i32 0
; ITERS3:  @specialized.arg.2 = internal constant i32 6
; ITERS3:  @specialized.arg.3 = internal constant i32 1
; ITERS3:  @specialized.arg.4 = internal constant i32 5
; ITERS3:  @specialized.arg.5 = internal constant i32 2
; ITERS3:  @specialized.arg.6 = internal constant i32 4
; ITERS3:  @specialized.arg.7 = internal constant i32 3
; ITERS3:  @specialized.arg.8 = internal constant i32 3

; ITERS4:  @specialized.arg.1 = internal constant i32 0
; ITERS4:  @specialized.arg.2 = internal constant i32 6
; ITERS4:  @specialized.arg.3 = internal constant i32 1
; ITERS4:  @specialized.arg.4 = internal constant i32 5
; ITERS4:  @specialized.arg.5 = internal constant i32 2
; ITERS4:  @specialized.arg.6 = internal constant i32 4
; ITERS4:  @specialized.arg.7 = internal constant i32 3
; ITERS4:  @specialized.arg.8 = internal constant i32 3

define i32 @main() {
; ITERS1-LABEL: @main(
; ITERS1-NEXT:    call void @print_val(i32 0, i32 6)
; ITERS1-NEXT:    call void @recursiveFunc(ptr nonnull @specialized.arg.3, i32 1, ptr nonnull @specialized.arg.4)
; ITERS1-NEXT:    ret i32 0
;
; ITERS2-LABEL: @main(
; ITERS2-NEXT:    call void @print_val(i32 0, i32 6)
; ITERS2-NEXT:    call void @print_val(i32 1, i32 5)
; ITERS2-NEXT:    call void @recursiveFunc(ptr nonnull @specialized.arg.5, i32 1, ptr nonnull @specialized.arg.6)
; ITERS2-NEXT:    ret i32 0
;
; ITERS3-LABEL: @main(
; ITERS3-NEXT:    call void @print_val(i32 0, i32 6)
; ITERS3-NEXT:    call void @print_val(i32 1, i32 5)
; ITERS3-NEXT:    call void @print_val(i32 2, i32 4)
; ITERS3-NEXT:    call void @recursiveFunc(ptr nonnull @specialized.arg.7, i32 1, ptr nonnull @specialized.arg.8)
; ITERS3-NEXT:    ret i32 0
;
; ITERS4-LABEL: @main(
; ITERS4-NEXT:    call void @print_val(i32 0, i32 6)
; ITERS4-NEXT:    call void @print_val(i32 1, i32 5)
; ITERS4-NEXT:    call void @print_val(i32 2, i32 4)
; ITERS4-NEXT:    ret i32 0
;
  %lo.temp = alloca i32, align 4
  %hi.temp = alloca i32, align 4
  %lo.load = load i32, ptr @low, align 4
  %hi.load = load i32, ptr @high, align 4
  store i32 %lo.load, ptr %lo.temp, align 4
  store i32 %hi.load, ptr %hi.temp, align 4
  call void @recursiveFunc(ptr nonnull %lo.temp, i32 1, ptr nonnull %hi.temp)
  ret i32 0
}

declare dso_local void @print_val(i32, i32)
