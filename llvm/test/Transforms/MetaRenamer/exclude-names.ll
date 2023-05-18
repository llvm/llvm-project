; RUN: opt -passes=metarenamer -rename-exclude-function-prefixes=my_func -rename-exclude-global-prefixes=my_global -rename-exclude-struct-prefixes=my_struct -rename-exclude-alias-prefixes=my_alias -S %s | FileCheck %s

; Check that excluded names don't get renamed while all the other ones do

; CHECK: %my_struct1 = type { ptr, i32 }
; CHECK: %my_struct2 = type { ptr, i32 }
; CHECK-NOT: %other_struct = type { ptr, i32 }
; CHECK: @my_global1 = global i32 42
; CHECK: @my_global2 = global i32 24
; CHECK-NOT: @other_global = global i32 24
; CHECK: @my_alias1 = alias i32, ptr @my_global1
; CHECK: @my_alias2 = alias i32, ptr @my_global2
; CHECK-NOT: @other_alias = alias i32, ptr @other_global
; CHECK: declare void @my_func1
; CHECK: declare void @my_func2
; CHECK-NOT: declare void @other_func

; CHECK: call void @my_func1
; CHECK: call void @my_func2
; CHECK-NOT: call void @other_func
; CHECK: load i32, ptr @my_global1
; CHECK: load i32, ptr @my_global2
; CHECK-NOT: load i32, ptr @other_global
; CHECK: load i32, ptr @my_alias1
; CHECK: load i32, ptr @my_alias2
; CHECK-NOT: load i32, ptr @other_alias
; CHECK: alloca %my_struct1
; CHECK: alloca %my_struct2
; CHECK-NOT: alloca %other_struct

%my_struct1 = type { ptr, i32 }
%my_struct2 = type { ptr, i32 }
%other_struct = type { ptr, i32 }
@my_global1 = global i32 42
@my_global2 = global i32 24
@other_global = global i32 24
@my_alias1 = alias i32, ptr @my_global1
@my_alias2 = alias i32, ptr @my_global2
@other_alias = alias i32, ptr @other_global
declare void @my_func1()
declare void @my_func2()
declare void @other_func()

define void @some_func() {
  call void @my_func1()
  call void @my_func2()
  call void @other_func()
  %a = load i32, ptr @my_global1
  %b = load i32, ptr @my_global2
  %c = load i32, ptr @other_global
  %d = load i32, ptr @my_alias1
  %e = load i32, ptr @my_alias2
  %f = load i32, ptr @other_alias
  %g = alloca %my_struct1
  %h = alloca %my_struct2
  %i = alloca %other_struct
  ret void
}
