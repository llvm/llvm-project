; RUN: llc %s -mtriple=x86_64-unknown-unknown -pass-remarks=stack-protector -o /dev/null 2>&1 | FileCheck %s
; CHECK-NOT: nossp
; CHECK: function attribute_ssp
; CHECK-SAME: a function attribute or command-line switch
; CHECK-NOT: alloca_fixed_small_nossp
; CHECK: function alloca_fixed_small_ssp
; CHECK-SAME: a call to alloca or use of a variable length array
; CHECK: function alloca_fixed_large_ssp
; CHECK-SAME: a call to alloca or use of a variable length array
; CHECK: function alloca_variable_ssp
; CHECK-SAME: a call to alloca or use of a variable length array
; CHECK: function buffer_ssp
; CHECK-SAME: a stack allocated buffer or struct containing a buffer
; CHECK: function struct_ssp
; CHECK-SAME: a stack allocated buffer or struct containing a buffer
; CHECK: function address_ssp
; CHECK-SAME: the address of a local variable being taken
; CHECK: function multiple_ssp
; CHECK-SAME: a function attribute or command-line switch
; CHECK: function multiple_ssp
; CHECK-SAME: a stack allocated buffer or struct containing a buffer
; CHECK: function multiple_ssp
; CHECK-SAME: a stack allocated buffer or struct containing a buffer
; CHECK: function multiple_ssp
; CHECK-SAME: the address of a local variable being taken
; CHECK: function multiple_ssp
; CHECK-SAME: a call to alloca or use of a variable length array

; Check that no remark is emitted when the switch is not specified.
; RUN: llc %s -mtriple=x86_64-unknown-unknown -o /dev/null 2>&1 | FileCheck %s -check-prefix=NOREMARK -allow-empty
; NOREMARK-NOT: ssp

define void @nossp() ssp {
  ret void
}

define void @attribute_ssp() sspreq {
  ret void
}

define void @alloca_fixed_small_nossp() ssp {
  %1 = alloca i8, i64 2, align 16
  ret void
}

define void @alloca_fixed_small_ssp() sspstrong {
  %1 = alloca i8, i64 2, align 16
  ret void
}

define void @alloca_fixed_large_ssp() ssp {
  %1 = alloca i8, i64 64, align 16
  ret void
}

define void @alloca_variable_ssp(i64 %x) ssp {
  %1 = alloca i8, i64 %x, align 16
  ret void
}

define void @buffer_ssp() sspstrong {
  %x = alloca [64 x i32], align 16
  ret void
}

%struct.X = type { [64 x i32] }
define void @struct_ssp() sspstrong {
  %x = alloca %struct.X, align 4
  ret void
}

define void @address_ssp() sspstrong {
entry:
  %x = alloca i32, align 4
  %y = alloca i32*, align 8
  store i32 32, i32* %x, align 4
  store i32* %x, i32** %y, align 8
  ret void
}

define void @multiple_ssp() sspreq {
entry:
  %x = alloca %struct.X, align 4
  %y = alloca [64 x i32], align 16
  %a = alloca i32, align 4
  %b = alloca i32*, align 8
  %0 = alloca i8, i64 2, align 16
  store i32 32, i32* %a, align 4
  store i32* %a, i32** %b, align 8
  ret void
}
