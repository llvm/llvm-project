; RUN:  opt -hotcoldsplit -S < %s | FileCheck %s
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

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
  %inc = add nsw i32 %1, 1
  store i32 %inc, i32* @c, align 4
  %2 = load i32, i32* @c, align 4
  %inc1 = add nsw i32 %2, 1
  store i32 %inc1, i32* @c, align 4
  %3 = load i32, i32* @c, align 4
  %inc2 = add nsw i32 %3, 1
  store i32 %inc2, i32* @c, align 4
  %4 = load i32, i32* @c, align 4
  %inc3 = add nsw i32 %4, 1
  store i32 %inc3, i32* @c, align 4
  %5 = load i32, i32* @c, align 4
  %dec = add nsw i32 %5, -1
  store i32 %dec, i32* @c, align 4
  %6 = load i32, i32* @c, align 4
  %dec4 = add nsw i32 %6, -1
  store i32 %dec4, i32* @c, align 4
  %7 = load i32, i32* @c, align 4
  %inc5 = add nsw i32 %7, 1
  store i32 %inc5, i32* @c, align 4
  %8 = load i32, i32* @c, align 4
  %inc6 = add nsw i32 %8, 1
  store i32 %inc6, i32* @c, align 4
  %9 = load i32, i32* @c, align 4
  %add = add nsw i32 %9, 1
  store i32 %add, i32* %i, align 4
  %10 = load i32, i32* %i, align 4
  %sub = sub nsw i32 %10, 1
  store i32 %sub, i32* %j, align 4
  %11 = load i32, i32* %i, align 4
  %add7 = add nsw i32 %11, 2
  store i32 %add7, i32* %k, align 4
  call void @llvm.eh.sjlj.longjmp(i8* bitcast ([20 x i8*]* @buf to i8*))
  unreachable
}

declare void @llvm.eh.sjlj.longjmp(i8*) #1

; CHECK-LABEL: @main
; CHECK-NOT: main.cold.1
define dso_local i32 @main() #0 {
entry:
  %retval = alloca i32, align 4
  %i = alloca i32, align 4
  store i32 0, i32* %retval, align 4
  store i32 0, i32* %i, align 4
  %0 = call i8* @llvm.frameaddress.p0i8(i32 0)
  store i8* %0, i8** getelementptr inbounds ([20 x i8*], [20 x i8*]* @buf, i64 0, i64 0), align 16
  %1 = call i8* @llvm.stacksave()
  store i8* %1, i8** getelementptr inbounds ([20 x i8*], [20 x i8*]* @buf, i64 0, i64 2), align 16
  %2 = call i32 @llvm.eh.sjlj.setjmp(i8* bitcast ([20 x i8*]* @buf to i8*))
  %tobool = icmp ne i32 %2, 0
  br i1 %tobool, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  store i32 1, i32* %retval, align 4
  br label %return

if.end:                                           ; preds = %entry
  call void @f()
  store i32 0, i32* %retval, align 4
  br label %return

return:                                           ; preds = %if.end, %if.then
  %3 = load i32, i32* %retval, align 4
  ret i32 %3
}

declare i8* @llvm.frameaddress.p0i8(i32 immarg) #2

declare i8* @llvm.stacksave() #3

declare i32 @llvm.eh.sjlj.setjmp(i8*) #3

attributes #0 = { nounwind uwtable }
attributes #1 = { noreturn nounwind }
attributes #2 = { nounwind readnone }
attributes #3 = { nounwind }


