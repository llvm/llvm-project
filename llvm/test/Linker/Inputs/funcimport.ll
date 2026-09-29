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
