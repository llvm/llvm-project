; RUN: opt -safe-stack -S -mtriple=i386-pc-linux-gnu < %s -o - | FileCheck %s
; RUN: opt -safe-stack -S -mtriple=x86_64-pc-linux-gnu < %s -o - | FileCheck %s

%struct.deep = type { %union.anon }
%union.anon = type { %struct.anon }
%struct.anon = type { %struct.anon.0 }
%struct.anon.0 = type { %union.anon.1 }
%union.anon.1 = type { [2 x i8] }

@.str = private unnamed_addr constant [4 x i8] c"%s\0A\00", align 1

; [2 x i8] nested in several layers of structs and unions
;  safestack attribute
; Requires no protector.
; CHECK-LABEL: @foo(
define signext i8 @foo() nounwind uwtable safestack {
entry:
  ; CHECK-NOT: __safestack_unsafe_stack_ptr
  %x = alloca %struct.deep, align 1
  %0 = load i8, ptr %x, align 1
  ret i8 %0
}
