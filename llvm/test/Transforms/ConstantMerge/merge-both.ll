; RUN: opt -constmerge %s -S -o - | FileCheck %s
; Test that in one run var3 is merged into var2 and var1 into var4.
; Test that we merge @var5 and @var6 into one with the higher alignment, and
; don't merge var7/var8 into var5/var6.

declare void @zed(%struct.foobar*, %struct.foobar*)

%struct.foobar = type { i32 }

@var1 = internal constant %struct.foobar { i32 2 }
@var2 = unnamed_addr constant %struct.foobar { i32 2 }
@var3 = internal constant %struct.foobar { i32 2 }
@var4 = unnamed_addr constant %struct.foobar { i32 2 }

; CHECK:      %struct.foobar = type { i32 }
; CHECK-NOT: @
; CHECK: @var2 = constant %struct.foobar { i32 2 }
; CHECK-NEXT: @var4 = constant %struct.foobar { i32 2 }

declare void @helper([16 x i8]*)
@var5 = internal constant [16 x i8] c"foo1bar2foo3bar\00", align 16
@var6 = private unnamed_addr constant [16 x i8] c"foo1bar2foo3bar\00", align 1
@var7 = internal constant [16 x i8] c"foo1bar2foo3bar\00"
@var8 = private unnamed_addr constant [16 x i8] c"foo1bar2foo3bar\00"

; CHECK-NEXT: @var6 = private constant [16 x i8] c"foo1bar2foo3bar\00", align 16
; CHECK-NEXT: @var8 = private constant [16 x i8] c"foo1bar2foo3bar\00"

define i32 @main() {
entry:
  call void @zed(%struct.foobar* @var1, %struct.foobar* @var2)
  call void @zed(%struct.foobar* @var3, %struct.foobar* @var4)
  call void @helper([16 x i8]* @var5)
  call void @helper([16 x i8]* @var6)
  call void @helper([16 x i8]* @var7)
  call void @helper([16 x i8]* @var8)
  ret i32 0
}

