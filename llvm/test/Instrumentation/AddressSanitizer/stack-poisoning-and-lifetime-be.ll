; Regular stack poisoning.
; RUN: opt < %s -passes=asan -asan-use-after-scope=0 -S | FileCheck --check-prefixes=CHECK,ENTRY,EXIT %s

; Stack poisoning with stack-use-after-scope.
; RUN: opt < %s -passes=asan -asan-use-after-scope=1 -S | FileCheck --check-prefixes=CHECK,ENTRY-UAS,EXIT-UAS %s

target datalayout = "E-m:e-i64:64-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

declare void @Foo(ptr)

define void @Bar() uwtable sanitize_address {
entry:
  %x = alloca [650 x i8], align 16
  %xx = getelementptr inbounds [650 x i8], ptr %x, i64 0, i64 0

  %y = alloca [13 x i8], align 1
  %yy = getelementptr inbounds [13 x i8], ptr %y, i64 0, i64 0

  %z = alloca [40 x i8], align 1
  %zz = getelementptr inbounds [40 x i8], ptr %z, i64 0, i64 0

  ; CHECK: [[SHADOW_BASE:%[0-9]+]] = add i64 %{{[0-9]+}}, 17592186044416

  ; F1F1F1F1
  ; ENTRY-NEXT: [[OFFSET:%[0-9]+]] = add i64 [[SHADOW_BASE]], 0
  ; ENTRY-NEXT: [[PTR:%[0-9]+]] = inttoptr i64 [[OFFSET]] to ptr
  ; ENTRY-NEXT: store i32 -235802127, ptr [[PTR]], align 1

  ; 02F2F2F2F2F2F2F2
  ; ENTRY-NEXT: [[OFFSET:%[0-9]+]] = add i64 [[SHADOW_BASE]], 85
  ; ENTRY-NEXT: [[PTR:%[0-9]+]] = inttoptr i64 [[OFFSET]] to ptr
  ; ENTRY-NEXT: store i64 212499257711850226, ptr [[PTR]], align 1

  ; F2F2F2F2F2F2F2F2
  ; ENTRY-NEXT: [[OFFSET:%[0-9]+]] = add i64 [[SHADOW_BASE]], 93
  ; ENTRY-NEXT: [[PTR:%[0-9]+]] = inttoptr i64 [[OFFSET]] to ptr
  ; ENTRY-NEXT: store i64 -940422246894996750, ptr [[PTR]], align 1

  ; F20005F2F2000000
  ; ENTRY-NEXT: [[OFFSET:%[0-9]+]] = add i64 [[SHADOW_BASE]], 101
  ; ENTRY-NEXT: [[PTR:%[0-9]+]] = inttoptr i64 [[OFFSET]] to ptr
  ; ENTRY-NEXT: store i64 -1008799775530680320, ptr [[PTR]], align 1

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
  ; ENTRY-UAS-NEXT: store i64 -506387832706107144, ptr [[PTR]], align 1

  ; F8F3F3F3
  ; ENTRY-UAS-NEXT: [[OFFSET:%[0-9]+]] = add i64 [[SHADOW_BASE]], 110
  ; ENTRY-UAS-NEXT: [[PTR:%[0-9]+]] = inttoptr i64 [[OFFSET]] to ptr
  ; ENTRY-UAS-NEXT: store i32 -118230029, ptr [[PTR]], align 1

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
  ; ENTRY-UAS-NEXT: store i16 5, ptr [[PTR]], align 1

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

declare void @llvm.lifetime.start.p0(i64, ptr nocapture)
declare void @llvm.lifetime.end.p0(i64, ptr nocapture)

; CHECK-ON: declare void @__asan_set_shadow_00(i64, i64)
; CHECK-ON: declare void @__asan_set_shadow_f1(i64, i64)
; CHECK-ON: declare void @__asan_set_shadow_f2(i64, i64)
; CHECK-ON: declare void @__asan_set_shadow_f3(i64, i64)
; CHECK-ON: declare void @__asan_set_shadow_f5(i64, i64)
; CHECK-ON: declare void @__asan_set_shadow_f8(i64, i64)

; CHECK-OFF-NOT: declare void @__asan_set_shadow_
