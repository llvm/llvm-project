; RUN:  opt -hotcoldsplit -S < %s | FileCheck %s
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.__jmp_buf_tag = type { [8 x i64], i32, %struct.__sigset_t }
%struct.__sigset_t = type { [16 x i64] }

@c = dso_local global i32 1, align 4
@buf = dso_local global [20 x i8*] zeroinitializer, align 16

; CHECK-LABEL: @f
; CHECK-NOT: f.cold.1
define dso_local void @f() #0 {
entry:
  %i = alloca i32, align 4
  %j = alloca i32, align 4
  %k = alloca i32, align 4
  %0 = load i32, i32* @c, align 4
  %tobool = icmp ne i32 %0, 0
  br i1 %tobool, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  ret void

if.else:                                          ; preds = %entry
  %1 = load i32, i32* @c, align 4
  %inc = add  i32 %1, 1
  store i32 %inc, i32* @c, align 4
  %2 = load i32, i32* @c, align 4
  %inc1 = add  i32 %2, 1
  store i32 %inc1, i32* @c, align 4
  %3 = load i32, i32* @c, align 4
  %inc2 = add  i32 %3, 1
  store i32 %inc2, i32* @c, align 4
  %4 = load i32, i32* @c, align 4
  %inc3 = add  i32 %4, 1
  store i32 %inc3, i32* @c, align 4
  %5 = load i32, i32* @c, align 4
  %dec = add  i32 %5, -1
  store i32 %dec, i32* @c, align 4
  %6 = load i32, i32* @c, align 4
  %dec4 = add  i32 %6, -1
  store i32 %dec4, i32* @c, align 4
  %7 = load i32, i32* @c, align 4
  %inc5 = add  i32 %7, 1
  store i32 %inc5, i32* @c, align 4
  %8 = load i32, i32* @c, align 4
  %inc6 = add  i32 %8, 1
  store i32 %inc6, i32* @c, align 4
  %9 = load i32, i32* @c, align 4
  %add = add  i32 %9, 1
  store i32 %add, i32* %i, align 4
  %10 = load i32, i32* %i, align 4
  %sub = sub  i32 %10, 1
  store i32 %sub, i32* %j, align 4
  %11 = load i32, i32* %i, align 4
  %add7 = add  i32 %11, 2
  store i32 %add7, i32* %k, align 4
  call void @longjmp(%struct.__jmp_buf_tag* bitcast ([20 x i8*]* @buf to %struct.__jmp_buf_tag*), i32 1) #3
  unreachable
}

declare dso_local void @longjmp(%struct.__jmp_buf_tag*, i32) #1

; CHECK-LABEL: @main
; CHECK-NOT: main.cold.1
define dso_local i32 @main() #0 {
entry:
  %retval = alloca i32, align 4
  %i = alloca i32, align 4
  store i32 0, i32* %retval, align 4
  store i32 0, i32* %i, align 4
  %call = call i32 @_setjmp(%struct.__jmp_buf_tag* bitcast ([20 x i8*]* @buf to %struct.__jmp_buf_tag*)) #4
  %tobool = icmp ne i32 %call, 0
  br i1 %tobool, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  store i32 1, i32* %retval, align 4
  br label %return

if.end:                                           ; preds = %entry
  call void @f()
  store i32 0, i32* %retval, align 4
  br label %return

return:                                           ; preds = %if.end, %if.then
  %0 = load i32, i32* %retval, align 4
  ret i32 %0
}

declare dso_local i32 @_setjmp(%struct.__jmp_buf_tag*) #2

attributes #0 = { nounwind uwtable }
attributes #1 = { noreturn nounwind }
attributes #2 = { nounwind returns_twice }
attributes #3 = { noreturn nounwind }
attributes #4 = { nounwind returns_twice }
