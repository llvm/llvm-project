target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.11.0"


define i32 @main() #0 {
entry:
  call void (...) @weakalias()
  call void (...) @analias()
  %call = call i32 (...) @referencestatics()
  %call1 = call i32 (...) @referenceglobals()
  %call2 = call i32 (...) @referencecommon()
  call void (...) @setfuncptr()
  call void (...) @callfuncptr()
  call void (...) @callweakfunc()
  ret i32 0
}

declare void @weakalias(...) #1

declare void @analias(...) #1

declare i32 @referencestatics(...) #1

declare i32 @referenceglobals(...) #1

declare i32 @referencecommon(...) #1

declare void @setfuncptr(...) #1

declare void @callfuncptr(...) #1

declare void @callweakfunc(...) #1
