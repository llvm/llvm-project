; Regular stack poisoning.
; RUN: opt < %s -passes=asan -asan-use-after-scope=0 -S | FileCheck --check-prefixes=CHECK,ENTRY,EXIT %s

; Stack poisoning with stack-use-after-scope.
; RUN: opt < %s -passes=asan -asan-use-after-scope=1 -S | FileCheck --check-prefixes=CHECK,ENTRY-UAS,EXIT-UAS %s

target datalayout = "e-i64:64-f80:128-s:64-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare void @Foo(ptr)

define void @Bar() uwtable sanitize_address {
entry:
  %x = alloca [650 x i8], align 16
  %xx = getelementptr inbounds [650 x i8], ptr %x, i64 0, i64 0

  %y = alloca [13 x i8], align 1
  %yy = getelementptr inbounds [13 x i8], ptr %y, i64 0, i64 0

  %z = alloca [40 x i8], align 1
  %zz = getelementptr inbounds [40 x i8], ptr %z, i64 0, i64 0

  ; CHECK: [[SHADOW_BASE:%[0-9]+]] = add i64 %{{[0-9]+}}, 2147450880

  ; F1F1F1F1
  ; ENTRY-NEXT: [[OFFSET:%[0-9]+]] = add i64 [[SHADOW_BASE]], 0
  ; ENTRY-NEXT: [[PTR:%[0-9]+]] = inttoptr i64 [[OFFSET]] to ptr
  ; ENTRY-NEXT: store i32 -235802127, ptr [[PTR]], align 1

  ; 02F2F2F2F2F2F2F2
  ; ENTRY-NEXT: [[OFFSET:%[0-9]+]] = add i64 [[SHADOW_BASE]], 85
  ; ENTRY-NEXT: [[PTR:%[0-9]+]] = inttoptr i64 [[OFFSET]] to ptr
  ; ENTRY-NEXT: store i64 -940422246894996990, ptr [[PTR]], align 1

  ; F2F2F2F2F2F2F2F2
  ; ENTRY-NEXT: [[OFFSET:%[0-9]+]] = add i64 [[SHADOW_BASE]], 93
  ; ENTRY-NEXT: [[PTR:%[0-9]+]] = inttoptr i64 [[OFFSET]] to ptr
  ; ENTRY-NEXT: store i64 -940422246894996750, ptr [[PTR]], align 1

  ; F20005F2F2000000
  ; ENTRY-NEXT: [[OFFSET:%[0-9]+]] = add i64 [[SHADOW_BASE]], 101
  ; ENTRY-NEXT: [[PTR:%[0-9]+]] = inttoptr i64 [[OFFSET]] to ptr
  ; ENTRY-NEXT: store i64 1043442499826, ptr [[PTR]], align 1

  ; F3F3F3F3
  ; ENTRY-NEXT: [[OFFSET:%[0-9]+]] = add i64 [[SHADOW_BASE]], 111
  ; ENTRY-NEXT: [[PTR:%[0-9]+]] = inttoptr i64 [[OFFSET]] to ptr
  ; ENTRY-NEXT: store i32 -202116109, ptr [[PTR]], align 1

  ; F3
  ; ENTRY-NEXT: [[OFFSET:%[0-9]+]] = add i64 [[SHADOW_BASE]], 115
  ; ENTRY-NEXT: [[PTR:%[0-9]+]] = inttoptr i64 [[OFFSET]] to ptr
  ; ENTRY-NEXT: store i8 -13, ptr [[PTR]], align 1

  ; F1F1F1F1
  ; ENTRY-UAS-NEXT: [[OFFSET:%[0-9]+]] = add i64 [[SHADOW_BASE]], 0
  ; ENTRY-UAS-NEXT: [[PTR:%[0-9]+]] = inttoptr i64 [[OFFSET]] to ptr
  ; ENTRY-UAS-NEXT: store i32 -235802127, ptr [[PTR]], align 1

  ; F8F8F8...
  ; ENTRY-UAS-NEXT: [[OFFSET:%[0-9]+]] = add i64 [[SHADOW_BASE]], 4
  ; ENTRY-UAS-NEXT: call void @__asan_set_shadow_f8(i64 [[OFFSET]], i64 82)

  ; F2F2F2F2F2F2F2F2
  ; ENTRY-UAS-NEXT: [[OFFSET:%[0-9]+]] = add i64 [[SHADOW_BASE]], 86
  ; ENTRY-UAS-NEXT: [[PTR:%[0-9]+]] = inttoptr i64 [[OFFSET]] to ptr
  ; ENTRY-UAS-NEXT: store i64 -940422246894996750, ptr [[PTR]], align 1

  ; F2F2F2F2F2F2F2F2
  ; ENTRY-UAS-NEXT: [[OFFSET:%[0-9]+]] = add i64 [[SHADOW_BASE]], 94
  ; ENTRY-UAS-NEXT: [[PTR:%[0-9]+]] = inttoptr i64 [[OFFSET]] to ptr
  ; ENTRY-UAS-NEXT: store i64 -940422246894996750, ptr [[PTR]], align 1

  ; F8F8F2F2F8F8F8F8
  ; ENTRY-UAS-NEXT: [[OFFSET:%[0-9]+]] = add i64 [[SHADOW_BASE]], 102
  ; ENTRY-UAS-NEXT: [[PTR:%[0-9]+]] = inttoptr i64 [[OFFSET]] to ptr
  ; ENTRY-UAS-NEXT: store i64 -506381209967593224, ptr [[PTR]], align 1

  ; F8F3F3F3
  ; ENTRY-UAS-NEXT: [[OFFSET:%[0-9]+]] = add i64 [[SHADOW_BASE]], 110
  ; ENTRY-UAS-NEXT: [[PTR:%[0-9]+]] = inttoptr i64 [[OFFSET]] to ptr
  ; ENTRY-UAS-NEXT: store i32 -202116104, ptr [[PTR]], align 1

  ; F3F3
  ; ENTRY-UAS-NEXT: [[OFFSET:%[0-9]+]] = add i64 [[SHADOW_BASE]], 114
  ; ENTRY-UAS-NEXT: [[PTR:%[0-9]+]] = inttoptr i64 [[OFFSET]] to ptr
  ; ENTRY-UAS-NEXT: store i16 -3085, ptr [[PTR]], align 1

  ; CHECK-LABEL: %xx = getelementptr inbounds
  ; CHECK-NEXT: %yy = getelementptr inbounds
  ; CHECK-NEXT: %zz = getelementptr inbounds


  call void @llvm.lifetime.start.p0(i64 650, ptr %xx)
  ; 0000...
  ; ENTRY-UAS-NEXT: [[OFFSET:%[0-9]+]] = add i64 [[SHADOW_BASE]], 4
  ; ENTRY-UAS-NEXT: call void @__asan_set_shadow_00(i64 [[OFFSET]], i64 81)
  ; 02
  ; ENTRY-UAS-NEXT: [[OFFSET:%[0-9]+]] = add i64 [[SHADOW_BASE]], 85
  ; ENTRY-UAS-NEXT: [[PTR:%[0-9]+]] = inttoptr i64 [[OFFSET]] to ptr
  ; ENTRY-UAS-NEXT: store i8 2, ptr [[PTR]], align 1

  ; CHECK-NEXT: call void @llvm.lifetime.start.p0(i64 650, ptr %xx)

  call void @Foo(ptr %xx)
  ; CHECK-NEXT: call void @Foo(ptr %xx)

  call void @llvm.lifetime.end.p0(i64 650, ptr %xx)
  ; ENTRY-UAS-NEXT: [[OFFSET:%[0-9]+]] = add i64 [[SHADOW_BASE]], 4
  ; ENTRY-UAS-NEXT: call void @__asan_set_shadow_f8(i64 [[OFFSET]], i64 82)

  ; CHECK-NEXT: call void @llvm.lifetime.end.p0(i64 650, ptr %xx)


  call void @llvm.lifetime.start.p0(i64 13, ptr %yy)
  ; 0005
  ; ENTRY-UAS-NEXT: [[OFFSET:%[0-9]+]] = add i64 [[SHADOW_BASE]], 102
  ; ENTRY-UAS-NEXT: [[PTR:%[0-9]+]] = inttoptr i64 [[OFFSET]] to ptr
  ; ENTRY-UAS-NEXT: store i16 1280, ptr [[PTR]], align 1

  ; CHECK-NEXT: call void @llvm.lifetime.start.p0(i64 13, ptr %yy)

  call void @Foo(ptr %yy)
  ; CHECK-NEXT: call void @Foo(ptr %yy)

  call void @llvm.lifetime.end.p0(i64 13, ptr %yy)
  ; F8F8
  ; ENTRY-UAS-NEXT: [[OFFSET:%[0-9]+]] = add i64 [[SHADOW_BASE]], 102
  ; ENTRY-UAS-NEXT: [[PTR:%[0-9]+]] = inttoptr i64 [[OFFSET]] to ptr
  ; ENTRY-UAS-NEXT: store i16 -1800, ptr [[PTR]], align 1

  ; CHECK-NEXT: call void @llvm.lifetime.end.p0(i64 13, ptr %yy)


  call void @llvm.lifetime.start.p0(i64 40, ptr %zz)
  ; 00000000
  ; ENTRY-UAS-NEXT: [[OFFSET:%[0-9]+]] = add i64 [[SHADOW_BASE]], 106
  ; ENTRY-UAS-NEXT: [[PTR:%[0-9]+]] = inttoptr i64 [[OFFSET]] to ptr
  ; ENTRY-UAS-NEXT: store i32 0, ptr [[PTR]], align 1
  ; 00
  ; ENTRY-UAS-NEXT: [[OFFSET:%[0-9]+]] = add i64 [[SHADOW_BASE]], 110
  ; ENTRY-UAS-NEXT: [[PTR:%[0-9]+]] = inttoptr i64 [[OFFSET]] to ptr
  ; ENTRY-UAS-NEXT: store i8 0, ptr [[PTR]], align 1

  ; CHECK-NEXT: call void @llvm.lifetime.start.p0(i64 40, ptr %zz)

  call void @Foo(ptr %zz)
  ; CHECK-NEXT: call void @Foo(ptr %zz)

  call void @llvm.lifetime.end.p0(i64 40, ptr %zz)
  ; F8F8F8F8
  ; ENTRY-UAS-NEXT: [[OFFSET:%[0-9]+]] = add i64 [[SHADOW_BASE]], 106
  ; ENTRY-UAS-NEXT: [[PTR:%[0-9]+]] = inttoptr i64 [[OFFSET]] to ptr
  ; ENTRY-UAS-NEXT: store i32 -117901064, ptr [[PTR]], align 1
  ; F8
  ; ENTRY-UAS-NEXT: [[OFFSET:%[0-9]+]] = add i64 [[SHADOW_BASE]], 110
  ; ENTRY-UAS-NEXT: [[PTR:%[0-9]+]] = inttoptr i64 [[OFFSET]] to ptr
  ; ENTRY-UAS-NEXT: store i8 -8, ptr [[PTR]], align 1

  ; CHECK-NEXT: call void @llvm.lifetime.end.p0(i64 40, ptr %zz)

  ; CHECK: {{^[0-9]+}}:

  ; CHECK-NEXT: [[OFFSET:%[0-9]+]] = add i64 [[SHADOW_BASE]], 0
  ; CHECK-NEXT: call void @__asan_set_shadow_f5(i64 [[OFFSET]], i64 128)

  ; CHECK-NOT: add i64 [[SHADOW_BASE]]

  ; CHECK: {{^[0-9]+}}:

  ; 00000000
  ; EXIT-NEXT: [[OFFSET:%[0-9]+]] = add i64 [[SHADOW_BASE]], 0
  ; EXIT-NEXT: [[PTR:%[0-9]+]] = inttoptr i64 [[OFFSET]] to ptr
  ; EXIT-NEXT: store i32 0, ptr [[PTR]], align 1

  ; 0000000000000000
  ; EXIT-NEXT: [[OFFSET:%[0-9]+]] = add i64 [[SHADOW_BASE]], 85
  ; EXIT-NEXT: [[PTR:%[0-9]+]] = inttoptr i64 [[OFFSET]] to ptr
  ; EXIT-NEXT: store i64 0, ptr [[PTR]], align 1

  ; 0000000000000000
  ; EXIT-NEXT: [[OFFSET:%[0-9]+]] = add i64 [[SHADOW_BASE]], 93
  ; EXIT-NEXT: [[PTR:%[0-9]+]] = inttoptr i64 [[OFFSET]] to ptr
  ; EXIT-NEXT: store i64 0, ptr [[PTR]], align 1

  ; 0000000000000000
  ; EXIT-NEXT: [[OFFSET:%[0-9]+]] = add i64 [[SHADOW_BASE]], 101
  ; EXIT-NEXT: [[PTR:%[0-9]+]] = inttoptr i64 [[OFFSET]] to ptr
  ; EXIT-NEXT: store i64 0, ptr [[PTR]], align 1

  ; 00000000
  ; EXIT-NEXT: [[OFFSET:%[0-9]+]] = add i64 [[SHADOW_BASE]], 111
  ; EXIT-NEXT: [[PTR:%[0-9]+]] = inttoptr i64 [[OFFSET]] to ptr
  ; EXIT-NEXT: store i32 0, ptr [[PTR]], align 1

  ; 00
  ; EXIT-NEXT: [[OFFSET:%[0-9]+]] = add i64 [[SHADOW_BASE]], 115
  ; EXIT-NEXT: [[PTR:%[0-9]+]] = inttoptr i64 [[OFFSET]] to ptr
  ; EXIT-NEXT: store i8 0, ptr [[PTR]], align 1

  ; 0000...
  ; EXIT-UAS-NEXT: [[OFFSET:%[0-9]+]] = add i64 [[SHADOW_BASE]], 0
  ; EXIT-UAS-NEXT: call void @__asan_set_shadow_00(i64 [[OFFSET]], i64 116)

  ; CHECK-NOT: add i64 [[SHADOW_BASE]]

  ret void
  ; CHECK: {{^[0-9]+}}:
  ; CHECK: ret void
}

declare void @foo(ptr)
define void @PR41481(i1 %b) sanitize_address {
; CHECK-LABEL: @PR41481
entry:
  %p1 = alloca i32
  %p2 = alloca i32
  br label %bb1

  ; Since we cannot account for all lifetime intrinsics in this function, we
  ; might have missed a lifetime.start one and therefore shouldn't poison the
  ; allocas at function entry.
  ; ENTRY: store i64 -935356719533264399
  ; ENTRY-UAS: store i64 -935356719533264399

bb1:
  %p = select i1 %b, ptr %p1, ptr %p2
  %q = select i1 %b, ptr  %p1, ptr  %p2
  call void @llvm.lifetime.start.p0(i64 4, ptr %q)
  call void @foo(ptr %p)
  br i1 %b, label %bb2, label %bb3

bb2:
  call void @llvm.lifetime.end.p0(i64 4, ptr %p1)
  br label %end

bb3:
  call void @llvm.lifetime.end.p0(i64 4, ptr %p2)
  br label %end

end:
  ret void
}


declare void @llvm.lifetime.start.p0(i64, ptr nocapture)
declare void @llvm.lifetime.end.p0(i64, ptr nocapture)

; CHECK-ON: declare void @__asan_set_shadow_00(i64, i64)
; CHECK-ON: declare void @__asan_set_shadow_f1(i64, i64)
; CHECK-ON: declare void @__asan_set_shadow_f2(i64, i64)
; CHECK-ON: declare void @__asan_set_shadow_f3(i64, i64)
; CHECK-ON: declare void @__asan_set_shadow_f5(i64, i64)
; CHECK-ON: declare void @__asan_set_shadow_f8(i64, i64)

; CHECK-OFF-NOT: declare void @__asan_set_shadow_
