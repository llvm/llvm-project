; RUN: llc < %s -mtriple=x86_64-apple-darwin | FileCheck %s
; <rdar://problem/10564621>

%0 = type opaque
%struct.NSConstantString = type { i32*, i32, i8*, i32 }

; Make sure that the string ends up the correct section.

; CHECK:        .section __TEXT,__cstring
; CHECK-NEXT: L_.str3:

; CHECK:        .section  __DATA,__cfstring
; CHECK-NEXT:   .p2align  4
; CHECK-NEXT: L__unnamed_cfstring_4:
; CHECK-NEXT:   .quad  ___CFConstantStringClassReference
; CHECK-NEXT:   .long  1992
; CHECK-NEXT:   .space  4
; CHECK-NEXT:   .quad  L_.str3
; CHECK-NEXT:   .long  0
; CHECK-NEXT:   .space  4

@isLogVisible = global i8 0, align 1
@__CFConstantStringClassReference = external global [0 x i32]
@.str3 = private unnamed_addr constant [1 x i8] zeroinitializer, align 1
@_unnamed_cfstring_4 = private constant %struct.NSConstantString { i32* getelementptr inbounds ([0 x i32], [0 x i32]* @__CFConstantStringClassReference, i32 0, i32 0), i32 1992, i8* getelementptr inbounds ([1 x i8], [1 x i8]* @.str3, i32 0, i32 0), i32 0 }, section "__DATA,__cfstring"
@null.array = weak_odr constant [1 x i8] zeroinitializer, align 1

define linkonce_odr void @bar() nounwind ssp align 2 {
entry:
  %stack = alloca i8*, align 4
  %call = call %0* bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to %0* (i8*, i8*, %0*)*)(i8* null, i8* null, %0* bitcast (%struct.NSConstantString* @_unnamed_cfstring_4 to %0*))
  store i8* getelementptr inbounds ([1 x i8], [1 x i8]* @null.array, i32 0, i32 0), i8** %stack, align 4
  ret void
}

declare i8* @objc_msgSend(i8*, i8*, ...) nonlazybind
