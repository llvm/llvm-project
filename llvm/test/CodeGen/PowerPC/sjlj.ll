; RUN: llc < %s -mtriple=powerpc64-unknown-linux-gnu -mcpu=pwr7 -verify-machineinstrs | FileCheck %s
; RUN: llc < %s -mtriple=powerpc64-unknown-linux-gnu -mcpu=a2 -verify-machineinstrs | FileCheck -check-prefix=CHECK-NOAV %s
target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-f128:128:128-v128:128:128-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

%struct.__jmp_buf_tag = type { [64 x i64], i32, %struct.__sigset_t, [8 x i8] }
%struct.__sigset_t = type { [16 x i64] }

@env_sigill = internal global [1 x %struct.__jmp_buf_tag] zeroinitializer, align 16
@cond = external global i8, align 1

define void @foo() #0 {
entry:
  call void @llvm.eh.sjlj.longjmp(i8* bitcast ([1 x %struct.__jmp_buf_tag]* @env_sigill to i8*))
  unreachable

; CHECK: @foo
; CHECK: addis [[REG:[0-9]+]], 2, env_sigill@toc@ha
; CHECK: addi [[REG]], [[REG]], env_sigill@toc@l
; CHECK: ld 31, 0([[REG]])
; CHECK: ld [[REG2:[0-9]+]], 8([[REG]])
; CHECK-DAG: ld 1, 16([[REG]])
; CHECK-DAG: mtctr [[REG2]]
; CHECK-DAG: ld 30, 32([[REG]])
; CHECK-DAG: ld 2, 24([[REG]])
; CHECK: bctr

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

; FIXME: We should be saving VRSAVE on Darwin, but we're not!

; CHECK-LABEL: main:
; CHECK: std
; Make sure that we're not saving VRSAVE on non-Darwin:
; CHECK-NOT: mfspr

; CHECK-DAG: stfd
; CHECK-DAG: stxvd2x

; CHECK-DAG: addis [[REG:[0-9]+]], 2, env_sigill@toc@ha
; CHECK-DAG: std 31, env_sigill@toc@l([[REG]])
; CHECK-DAG: addi [[REGA:[0-9]+]], [[REG]], env_sigill@toc@l
; CHECK-DAG: std [[REGA]], [[OFF:[0-9]+]](31)                  # 8-byte Folded Spill
; CHECK-DAG: std 1, 16([[REGA]])
; CHECK-DAG: std 2, 24([[REGA]])
; CHECK: bcl 20, 31, .LBB1_3
; CHECK: li 3, 1
; CHECK: #EH_SjLj_Setup	.LBB1_3
; CHECK: # %bb.1:

; CHECK: .LBB1_3:
; CHECK: mflr [[REGL:[0-9]+]]
; CHECK: ld [[REG2:[0-9]+]], [[OFF]](31)                   # 8-byte Folded Reload
; CHECK: std [[REGL]], 8([[REG2]])
; CHECK: li 3, 0

; CHECK: .LBB1_5:

; CHECK-DAG: lfd
; CHECK-DAG: lxvd2x
; CHECK: ld
; CHECK: blr

; CHECK-NOAV-LABEL: main:
; CHECK-NOAV-NOT: stxvd2x
; CHECK-NOAV: bcl
; CHECK-NOAV: mflr
; CHECK-NOAV: bl foo
; CHECK-NOAV-NOT: lxvd2x
; CHECK-NOAV: blr
}

define signext i32 @main2() #0 {
entry:
  %a = alloca i8, align 64
  call void @bar(i8* %a)
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

; CHECK-LABEL: main2:

; CHECK: addis [[REG:[0-9]+]], 2, env_sigill@toc@ha
; CHECK-DAG: std 31, env_sigill@toc@l([[REG]])
; CHECK-DAG: addi [[REGB:[0-9]+]], [[REG]], env_sigill@toc@l
; CHECK-DAG: std [[REGB]], [[OFF:[0-9]+]](31)                  # 8-byte Folded Spill
; CHECK-DAG: std 1, 16([[REGB]])
; CHECK-DAG: std 2, 24([[REGB]])
; CHECK-DAG: std 30, 32([[REGB]])
; CHECK: bcl 20, 31,

; CHECK: blr
}

define void @test_sjlj_setjmp() #0 {
entry:
  %0 = load i8, i8* @cond, align 1
  %tobool = trunc i8 %0 to i1
  br i1 %tobool, label %return, label %end

end:
  %1 = call i32 @llvm.eh.sjlj.setjmp(i8* bitcast ([1 x %struct.__jmp_buf_tag]* @env_sigill to i8*))
  br label %return

return:
  ret void

; CHECK-LABEL: test_sjlj_setjmp:
; intrinsic llvm.eh.sjlj.setjmp does not call builtin function _setjmp.
; CHECK-NOT: bl _setjmp
}

declare void @bar(i8*) #3

declare i8* @llvm.frameaddress(i32) #2

declare i8* @llvm.stacksave() #3

declare i32 @llvm.eh.sjlj.setjmp(i8*) #3

attributes #0 = { nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { noreturn nounwind }
attributes #2 = { nounwind readnone }
attributes #3 = { nounwind }

