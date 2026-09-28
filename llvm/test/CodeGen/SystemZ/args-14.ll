; RUN: llc < %s -mtriple=s390x-linux-gnu -argext-abi-check
; REQUIRES: asserts

; Test that it works to pass structs as outgoing call arguments when the
; NoExt attribute is given, either in the call instruction or in the
; prototype of the called function.
define void @caller() {
  call void @bar_Struct_32(i32 noext 123)
  call void @bar_Struct_16(i16 123)
  call void @bar_Struct_8(i8 noext 123)
  call void @bar_Struct_32_addrtaken(i32 noext 123)
  ret void
}

declare void @bar_Struct_32(i32 %Arg)
declare void @bar_Struct_16(i16 noext %Arg)
declare void @bar_Struct_8(i8 %Arg)
define internal void @bar_Struct_32_addrtaken(i32 noext %Arg) {
  ret void
}

; Test that it works to return values with the NoExt attribute.
define noext i8 @fun_NoExtRet_i8() {
  ret i8 -1
}

define noext i16 @fun_NoExtRet_i16() {
  ret i16 -1
}

define noext i32 @fun_NoExtRet_i32() {
  ret i32 -1
}

define internal noext i32 @fun_NoExtRet_i32_addrtaken() {
  ret i32 -1
}

declare void @ExtFun(ptr %FunPtr)
define void @foo() {
  call void @ExtFun(ptr @bar_Struct_32_addrtaken)
  call void @ExtFun(ptr @fun_NoExtRet_i32_addrtaken)
  ret void
}

; A function that is internal or has a non-C ABI is not checked for an
; extension attribute.
define void @caller_non_C_abi(ptr %fptr) {
  call void @foo_internal(i32 0)
  call fastcc void @foo_fastcc(i32 0)
  call fastcc void %fptr(i32 0)
  ret void
}

define internal void @foo_internal(i32 %Arg) {
  ret void
}

declare fastcc void @foo_fastcc(i32 %Arg)

define internal i8 @fun_NoRetAttr_internal() {
  ret i8 -1
}

define fastcc i8 @fun_NoRetAttr_fastcc() {
  ret i8 -1
}
