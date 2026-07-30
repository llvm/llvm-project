; RUN: llc < %s -mtriple=s390x-linux-gnu -argext-abi-check

; Test that it works to pass structs as outgoing call arguments when the
; NoExt attribute is given, either in the call instruction or in the
; prototype of the called function.
define void @caller() {
  call void @bar_Struct_32(i32 noext 123)
  call void @bar_Struct_16(i16 123)
  call void @bar_Struct_8(i8 noext 123)
  ret void
}

declare void @bar_Struct_32(i32 %Arg)
declare void @bar_Struct_16(i16 noext %Arg)
declare void @bar_Struct_8(i8 %Arg)

; Test that it works to return values with the NoExt attribute.
define noext i8 @callee_NoExtRet_i8() {
  ret i8 -1
}

define noext i16 @callee_NoExtRet_i16() {
  ret i16 -1
}

define noext i32 @callee_NoExtRet_i32() {
  ret i32 -1
}

; An internal function is not checked for an extension attribute.
define internal i32 @callee_NoExtRet_internal(i32 %Arg) {
  ret i32 %Arg
}

; A call to an internal function is ok without argument extension.
define void @caller_internal() {
  call i32 @callee_NoExtRet_internal(i32 0)
  ret void
}
