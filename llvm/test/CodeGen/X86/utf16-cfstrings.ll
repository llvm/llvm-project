; RUN: llc < %s -mtriple x86_64-apple-macosx10 | FileCheck %s
; <rdar://problem/10655949>

%0 = type opaque
%struct.NSConstantString = type { i32*, i32, i8*, i64 }

@__CFConstantStringClassReference = external global [0 x i32]
@.str = internal unnamed_addr constant [5 x i16] [i16 252, i16 98, i16 101, i16 114, i16 0], align 2
@_unnamed_cfstring_ = private constant %struct.NSConstantString { i32* getelementptr inbounds ([0 x i32], [0 x i32]* @__CFConstantStringClassReference, i32 0, i32 0), i32 2000, i8* bitcast ([5 x i16]* @.str to i8*), i64 4 }, section "__DATA,__cfstring"

; CHECK:         .section      __TEXT,__ustring
; CHECK-NEXT:    .p2align        1
; CHECK-NEXT: _.str:
; CHECK-NEXT:    .short  252     ## 0xfc
; CHECK-NEXT:    .short  98      ## 0x62
; CHECK-NEXT:    .short  101     ## 0x65
; CHECK-NEXT:    .short  114     ## 0x72
; CHECK-NEXT:    .short  0       ## 0x0

define i32 @main() uwtable ssp {
entry:
  %retval = alloca i32, align 4
  store i32 0, i32* %retval
  call void (%0*, ...) @NSLog(%0* bitcast (%struct.NSConstantString* @_unnamed_cfstring_ to %0*))
  ret i32 0
}

declare void @NSLog(%0*, ...)

!llvm.module.flags = !{!0, !1, !2, !3}

!0 = !{i32 1, !"Objective-C Version", i32 2}
!1 = !{i32 1, !"Objective-C Image Info Version", i32 0}
!2 = !{i32 1, !"Objective-C Image Info Section", !"__DATA, __objc_imageinfo, regular, no_dead_strip"}
!3 = !{i32 4, !"Objective-C Garbage Collection", i32 0}
