; RUN: opt -passes=globaldce -disable-output -print-changed=inst-quiet %s 2>&1 | FileCheck %s --check-prefix=NO-CHANGE --allow-empty
; RUN: opt -passes=inferattrs -disable-output -print-changed=inst-quiet %s 2>&1 | FileCheck %s --check-prefix=NO-CHANGE --allow-empty
; RUN: opt -passes=no-op-module -disable-output -print-changed=inst %s 2>&1 | FileCheck %s --check-prefix=SNAPSHOT

target triple = "x86_64-unknown-linux-gnu"

%0 = type { i32 }
%1 = type { i64 }

@0 = internal global %0 zeroinitializer
@1 = global i32 0
@2 = alias i32, ptr @1

declare i64 @strlen(ptr)
declare void @bar()

define ptr @global_ref() {
entry:
  ret ptr @1
}

define ptr @alias_ref() {
entry:
  ret ptr @2
}

define void @3() {
entry:
  ret void
}

define void @function_ref() {
entry:
  call void @3()
  ret void
}

define void @attribute_ref() {
entry:
  call void @bar() #0
  ret void
}

define i64 @type_ref() {
entry:
  %value = alloca %1
  %field = getelementptr %1, ptr %value, i32 0, i32 0
  %result = load i64, ptr %field
  ret i64 %result
}

attributes #0 = { nounwind }

; NO-CHANGE-NOT: IR Instruction Changes

; SNAPSHOT: + block#[[UNNAMED_FUNCTION_BLOCK:[0-9]+]] @<[[UNNAMED_FUNCTION:[0-9]+]]>:0
; SNAPSHOT: + inst#[[RET:[0-9]+]] @global_ref block#[[GLOBAL_BLOCK:[0-9]+]]:0   ret ptr @<[[GLOBAL:[0-9]+]]>
; SNAPSHOT: + inst#[[ALIAS_RET:[0-9]+]] @alias_ref block#[[ALIAS_BLOCK:[0-9]+]]:0   ret ptr @<[[ALIAS:[0-9]+]]>
; SNAPSHOT: + inst#[[FUNCTION_CALL:[0-9]+]] @function_ref block#[[FUNCTION_BLOCK:[0-9]+]]:0   call void @<[[UNNAMED_FUNCTION]]>()
; SNAPSHOT: + inst#[[CALL:[0-9]+]] @attribute_ref block#[[ATTR_BLOCK:[0-9]+]]:0   call void @bar() nounwind
; SNAPSHOT: + inst#[[ALLOCA:[0-9]+]] @type_ref block#[[TYPE_BLOCK:[0-9]+]]:0   %value = alloca %type<[[TYPE:[0-9]+]]>, align 8
; SNAPSHOT: + inst#[[GEP:[0-9]+]] @type_ref block#[[TYPE_BLOCK]]:1   %field = getelementptr %type<[[TYPE]]>, ptr %value, i32 0, i32 0
