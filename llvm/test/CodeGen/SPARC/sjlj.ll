; RUN: llc < %s -march=sparc | FileCheck %s
; RUN: llc < %s -march=sparc -mcpu=leon2 | FileCheck %s
; RUN: llc < %s -march=sparc -mcpu=leon3 | FileCheck %s
; RUN: llc < %s -march=sparc -mcpu=leon4 | FileCheck %s

%struct.__jmp_buf_tag = type { [64 x i64], i32, %struct.__sigset_t, [8 x i8] }
%struct.__sigset_t = type { [16 x i64] }

@env_sigill = internal global [1 x %struct.__jmp_buf_tag] zeroinitializer, align 16

define void @foo() #0 {
entry:
  call void @llvm.eh.sjlj.longjmp(i8* bitcast ([1 x %struct.__jmp_buf_tag]* @env_sigill to i8*))
  unreachable

; CHECK: @foo
; CHECK: ta   3
; CHECK: ld [%i0], %fp
; CHECK: ld [%i0+4], %i1
; CHECK: ld [%i0+8], %sp
; CHECK: jmp %i1
; CHECK: ld [%i0+12], %i7

return:                                           ; No predecessors!
  ret void
}

declare void @llvm.eh.sjlj.longjmp(i8*) #1

define signext i32 @main() #0 {
entry:
  %retval = alloca i32, align 4
  store i32 0, i32* %retval
  %0 = call i8* @llvm.frameaddress(i32 0)
  store i8* %0, i8** bitcast ([1 x %struct.__jmp_buf_tag]* @env_sigill to i8**)
  %1 = call i8* @llvm.stacksave()
  store i8* %1, i8** getelementptr (i8*, i8** bitcast ([1 x %struct.__jmp_buf_tag]* @env_sigill to i8**), i32 2)
  %2 = call i32 @llvm.eh.sjlj.setjmp(i8* bitcast ([1 x %struct.__jmp_buf_tag]* @env_sigill to i8*))
  %tobool = icmp ne i32 %2, 0
  br i1 %tobool, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  store i32 1, i32* %retval
  br label %return

if.else:                                          ; preds = %entry
  call void @foo()
  br label %if.end

if.end:                                           ; preds = %if.else
  store i32 0, i32* %retval
  br label %return

return:                                           ; preds = %if.end, %if.then
  %3 = load i32, i32* %retval
  ret i32 %3

; CHECK: @main
; CHECK:  st %fp, [%i0]
; CHECK:  sethi %hi(.LBB1_2), %i1
; CHECK:  or %i1, %lo(.LBB1_2), %i1
; CHECK:  st %i1, [%i0+4]
; CHECK:  st %sp, [%i0+8]
; CHECK:  bn   .LBB1_2
; CHECK:  st %i7, [%i0+12]
; CHECK:  ba   .LBB1_1
; CHECK:  nop
; CHECK:.LBB1_1:                                ! %entry
; CHECK:  mov  %g0, %i0
; CHECK:                                        ! %entry
; CHECK:  cmp %i0, 0
; CHECK:  be   .LBB1_5
; CHECK:  nop
; CHECK:.LBB1_4:
; CHECK:  mov  1, %i0
; CHECK:  ba .LBB1_6
; CHECK:.LBB1_2:                                ! Block address taken
; CHECK:  mov  1, %i0
; CHECK:  cmp %i0, 0
; CHECK:  bne  .LBB1_4
; CHECK:  nop
}
declare i8* @llvm.frameaddress(i32) #2

declare i8* @llvm.stacksave() #3

declare i32 @llvm.eh.sjlj.setjmp(i8*) #3

attributes #0 = { nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { noreturn nounwind }
attributes #2 = { nounwind readnone }
attributes #3 = { nounwind }

