; Regular stack poisoning.
; RUN: opt < %s -passes=asan -asan-use-after-scope=0 -S | FileCheck --check-prefixes=CHECK,ENTRY,EXIT %s

; Stack poisoning with stack-use-after-scope.
; RUN: opt < %s -passes=asan -asan-use-after-scope=1 -S | FileCheck --check-prefixes=CHECK,ENTRY-UAS,EXIT-UAS %s

target datalayout = "e-i64:64-f80:128-s:64-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare void @Foo(i8*)

define void @Bar() uwtable sanitize_address {
entry:
  %x = alloca [650 x i8], align 16
  %xx = getelementptr inbounds [650 x i8], [650 x i8]* %x, i64 0, i64 0

  %y = alloca [13 x i8], align 1
  %yy = getelementptr inbounds [13 x i8], [13 x i8]* %y, i64 0, i64 0

  %z = alloca [40 x i8], align 1
  %zz = getelementptr inbounds [40 x i8], [40 x i8]* %z, i64 0, i64 0

  ; CHECK: [[SHADOW_BASE:%[0-9]+]] = add i64 %{{[0-9]+}}, 2147450880

  ; F1F1F1F1
  ; ENTRY-NEXT: [[OFFSET:%[0-9]+]] = add i64 [[SHADOW_BASE]], 0
  ; ENTRY-NEXT: [[PTR:%[0-9]+]] = inttoptr i64 [[OFFSET]] to [[TYPE:i32]]*
  ; ENTRY-NEXT: store [[TYPE]] -235802127, [[TYPE]]* [[PTR]], align 1

  ; 02F2F2F2F2F2F2F2
  ; ENTRY-NEXT: [[OFFSET:%[0-9]+]] = add i64 [[SHADOW_BASE]], 85
  ; ENTRY-NEXT: [[PTR:%[0-9]+]] = inttoptr i64 [[OFFSET]] to [[TYPE:i64]]*
  ; ENTRY-NEXT: store [[TYPE]] -940422246894996990, [[TYPE]]* [[PTR]], align 1

  ; F2F2F2F2F2F2F2F2
  ; ENTRY-NEXT: [[OFFSET:%[0-9]+]] = add i64 [[SHADOW_BASE]], 93
  ; ENTRY-NEXT: [[PTR:%[0-9]+]] = inttoptr i64 [[OFFSET]] to [[TYPE:i64]]*
  ; ENTRY-NEXT: store [[TYPE]] -940422246894996750, [[TYPE]]* [[PTR]], align 1

  ; F20005F2F2000000
  ; ENTRY-NEXT: [[OFFSET:%[0-9]+]] = add i64 [[SHADOW_BASE]], 101
  ; ENTRY-NEXT: [[PTR:%[0-9]+]] = inttoptr i64 [[OFFSET]] to [[TYPE:i64]]*
  ; ENTRY-NEXT: store [[TYPE]] 1043442499826, [[TYPE]]* [[PTR]], align 1

  ; F3F3F3F3
  ; ENTRY-NEXT: [[OFFSET:%[0-9]+]] = add i64 [[SHADOW_BASE]], 111
  ; ENTRY-NEXT: [[PTR:%[0-9]+]] = inttoptr i64 [[OFFSET]] to [[TYPE:i32]]*
  ; ENTRY-NEXT: store [[TYPE]] -202116109, [[TYPE]]* [[PTR]], align 1

  ; F3
  ; ENTRY-NEXT: [[OFFSET:%[0-9]+]] = add i64 [[SHADOW_BASE]], 115
  ; ENTRY-NEXT: [[PTR:%[0-9]+]] = inttoptr i64 [[OFFSET]] to [[TYPE:i8]]*
  ; ENTRY-NEXT: store [[TYPE]] -13, [[TYPE]]* [[PTR]], align 1

  ; F1F1F1F1
  ; ENTRY-UAS-NEXT: [[OFFSET:%[0-9]+]] = add i64 [[SHADOW_BASE]], 0
  ; ENTRY-UAS-NEXT: [[PTR:%[0-9]+]] = inttoptr i64 [[OFFSET]] to [[TYPE:i32]]*
  ; ENTRY-UAS-NEXT: store [[TYPE]] -235802127, [[TYPE]]* [[PTR]], align 1

  ; F8F8F8...
  ; ENTRY-UAS-NEXT: [[OFFSET:%[0-9]+]] = add i64 [[SHADOW_BASE]], 4
  ; ENTRY-UAS-NEXT: call void @__asan_set_shadow_f8(i64 [[OFFSET]], i64 82)

  ; F2F2F2F2F2F2F2F2
  ; ENTRY-UAS-NEXT: [[OFFSET:%[0-9]+]] = add i64 [[SHADOW_BASE]], 86
  ; ENTRY-UAS-NEXT: [[PTR:%[0-9]+]] = inttoptr i64 [[OFFSET]] to [[TYPE:i64]]*
  ; ENTRY-UAS-NEXT: store [[TYPE]] -940422246894996750, [[TYPE]]* [[PTR]], align 1

  ; F2F2F2F2F2F2F2F2
  ; ENTRY-UAS-NEXT: [[OFFSET:%[0-9]+]] = add i64 [[SHADOW_BASE]], 94
  ; ENTRY-UAS-NEXT: [[PTR:%[0-9]+]] = inttoptr i64 [[OFFSET]] to [[TYPE:i64]]*
  ; ENTRY-UAS-NEXT: store [[TYPE]] -940422246894996750, [[TYPE]]* [[PTR]], align 1

  ; F8F8F2F2F8F8F8F8
  ; ENTRY-UAS-NEXT: [[OFFSET:%[0-9]+]] = add i64 [[SHADOW_BASE]], 102
  ; ENTRY-UAS-NEXT: [[PTR:%[0-9]+]] = inttoptr i64 [[OFFSET]] to [[TYPE:i64]]*
  ; ENTRY-UAS-NEXT: store [[TYPE]] -506381209967593224, [[TYPE]]* [[PTR]], align 1

  ; F8F3F3F3
  ; ENTRY-UAS-NEXT: [[OFFSET:%[0-9]+]] = add i64 [[SHADOW_BASE]], 110
  ; ENTRY-UAS-NEXT: [[PTR:%[0-9]+]] = inttoptr i64 [[OFFSET]] to [[TYPE:i32]]*
  ; ENTRY-UAS-NEXT: store [[TYPE]] -202116104, [[TYPE]]* [[PTR]], align 1

  ; F3F3
  ; ENTRY-UAS-NEXT: [[OFFSET:%[0-9]+]] = add i64 [[SHADOW_BASE]], 114
  ; ENTRY-UAS-NEXT: [[PTR:%[0-9]+]] = inttoptr i64 [[OFFSET]] to [[TYPE:i16]]*
  ; ENTRY-UAS-NEXT: store [[TYPE]] -3085, [[TYPE]]* [[PTR]], align 1

  ; CHECK-LABEL: %xx = getelementptr inbounds
  ; CHECK-NEXT: %yy = getelementptr inbounds
  ; CHECK-NEXT: %zz = getelementptr inbounds


  call void @llvm.lifetime.start.p0i8(i64 650, i8* %xx)
  ; 0000...
  ; ENTRY-UAS-NEXT: [[OFFSET:%[0-9]+]] = add i64 [[SHADOW_BASE]], 4
  ; ENTRY-UAS-NEXT: call void @__asan_set_shadow_00(i64 [[OFFSET]], i64 81)
  ; 02
  ; ENTRY-UAS-NEXT: [[OFFSET:%[0-9]+]] = add i64 [[SHADOW_BASE]], 85
  ; ENTRY-UAS-NEXT: [[PTR:%[0-9]+]] = inttoptr i64 [[OFFSET]] to [[TYPE:i8]]*
  ; ENTRY-UAS-NEXT: store [[TYPE]] 2, [[TYPE]]* [[PTR]], align 1

  ; CHECK-NEXT: call void @llvm.lifetime.start.p0i8(i64 650, i8* %xx)

  call void @Foo(i8* %xx)
  ; CHECK-NEXT: call void @Foo(i8* %xx)

  call void @llvm.lifetime.end.p0i8(i64 650, i8* %xx)
  ; ENTRY-UAS-NEXT: [[OFFSET:%[0-9]+]] = add i64 [[SHADOW_BASE]], 4
  ; ENTRY-UAS-NEXT: call void @__asan_set_shadow_f8(i64 [[OFFSET]], i64 82)

  ; CHECK-NEXT: call void @llvm.lifetime.end.p0i8(i64 650, i8* %xx)


  call void @llvm.lifetime.start.p0i8(i64 13, i8* %yy)
  ; 0005
  ; ENTRY-UAS-NEXT: [[OFFSET:%[0-9]+]] = add i64 [[SHADOW_BASE]], 102
  ; ENTRY-UAS-NEXT: [[PTR:%[0-9]+]] = inttoptr i64 [[OFFSET]] to [[TYPE:i16]]*
  ; ENTRY-UAS-NEXT: store [[TYPE]] 1280, [[TYPE]]* [[PTR]], align 1

  ; CHECK-NEXT: call void @llvm.lifetime.start.p0i8(i64 13, i8* %yy)

  call void @Foo(i8* %yy)
  ; CHECK-NEXT: call void @Foo(i8* %yy)

  call void @llvm.lifetime.end.p0i8(i64 13, i8* %yy)
  ; F8F8
  ; ENTRY-UAS-NEXT: [[OFFSET:%[0-9]+]] = add i64 [[SHADOW_BASE]], 102
  ; ENTRY-UAS-NEXT: [[PTR:%[0-9]+]] = inttoptr i64 [[OFFSET]] to [[TYPE:i16]]*
  ; ENTRY-UAS-NEXT: store [[TYPE]] -1800, [[TYPE]]* [[PTR]], align 1

  ; CHECK-NEXT: call void @llvm.lifetime.end.p0i8(i64 13, i8* %yy)


  call void @llvm.lifetime.start.p0i8(i64 40, i8* %zz)
  ; 00000000
  ; ENTRY-UAS-NEXT: [[OFFSET:%[0-9]+]] = add i64 [[SHADOW_BASE]], 106
  ; ENTRY-UAS-NEXT: [[PTR:%[0-9]+]] = inttoptr i64 [[OFFSET]] to [[TYPE:i32]]*
  ; ENTRY-UAS-NEXT: store [[TYPE]] 0, [[TYPE]]* [[PTR]], align 1
  ; 00
  ; ENTRY-UAS-NEXT: [[OFFSET:%[0-9]+]] = add i64 [[SHADOW_BASE]], 110
  ; ENTRY-UAS-NEXT: [[PTR:%[0-9]+]] = inttoptr i64 [[OFFSET]] to [[TYPE:i8]]*
  ; ENTRY-UAS-NEXT: store [[TYPE]] 0, [[TYPE]]* [[PTR]], align 1

  ; CHECK-NEXT: call void @llvm.lifetime.start.p0i8(i64 40, i8* %zz)

  call void @Foo(i8* %zz)
  ; CHECK-NEXT: call void @Foo(i8* %zz)

  call void @llvm.lifetime.end.p0i8(i64 40, i8* %zz)
  ; F8F8F8F8
  ; ENTRY-UAS-NEXT: [[OFFSET:%[0-9]+]] = add i64 [[SHADOW_BASE]], 106
  ; ENTRY-UAS-NEXT: [[PTR:%[0-9]+]] = inttoptr i64 [[OFFSET]] to [[TYPE:i32]]*
  ; ENTRY-UAS-NEXT: store [[TYPE]] -117901064, [[TYPE]]* [[PTR]], align 1
  ; F8
  ; ENTRY-UAS-NEXT: [[OFFSET:%[0-9]+]] = add i64 [[SHADOW_BASE]], 110
  ; ENTRY-UAS-NEXT: [[PTR:%[0-9]+]] = inttoptr i64 [[OFFSET]] to [[TYPE:i8]]*
  ; ENTRY-UAS-NEXT: store [[TYPE]] -8, [[TYPE]]* [[PTR]], align 1

  ; CHECK-NEXT: call void @llvm.lifetime.end.p0i8(i64 40, i8* %zz)

  ; CHECK: {{^[0-9]+}}:

  ; CHECK-NEXT: [[OFFSET:%[0-9]+]] = add i64 [[SHADOW_BASE]], 0
  ; CHECK-NEXT: call void @__asan_set_shadow_f5(i64 [[OFFSET]], i64 128)

  ; CHECK-NOT: add i64 [[SHADOW_BASE]]

  ; CHECK: {{^[0-9]+}}:

  ; 00000000
  ; EXIT-NEXT: [[OFFSET:%[0-9]+]] = add i64 [[SHADOW_BASE]], 0
  ; EXIT-NEXT: [[PTR:%[0-9]+]] = inttoptr i64 [[OFFSET]] to [[TYPE:i32]]*
  ; EXIT-NEXT: store [[TYPE]] 0, [[TYPE]]* [[PTR]], align 1

  ; 0000000000000000
  ; EXIT-NEXT: [[OFFSET:%[0-9]+]] = add i64 [[SHADOW_BASE]], 85
  ; EXIT-NEXT: [[PTR:%[0-9]+]] = inttoptr i64 [[OFFSET]] to [[TYPE:i64]]*
  ; EXIT-NEXT: store [[TYPE]] 0, [[TYPE]]* [[PTR]], align 1

  ; 0000000000000000
  ; EXIT-NEXT: [[OFFSET:%[0-9]+]] = add i64 [[SHADOW_BASE]], 93
  ; EXIT-NEXT: [[PTR:%[0-9]+]] = inttoptr i64 [[OFFSET]] to [[TYPE:i64]]*
  ; EXIT-NEXT: store [[TYPE]] 0, [[TYPE]]* [[PTR]], align 1

  ; 0000000000000000
  ; EXIT-NEXT: [[OFFSET:%[0-9]+]] = add i64 [[SHADOW_BASE]], 101
  ; EXIT-NEXT: [[PTR:%[0-9]+]] = inttoptr i64 [[OFFSET]] to [[TYPE:i64]]*
  ; EXIT-NEXT: store [[TYPE]] 0, [[TYPE]]* [[PTR]], align 1

  ; 00000000
  ; EXIT-NEXT: [[OFFSET:%[0-9]+]] = add i64 [[SHADOW_BASE]], 111
  ; EXIT-NEXT: [[PTR:%[0-9]+]] = inttoptr i64 [[OFFSET]] to [[TYPE:i32]]*
  ; EXIT-NEXT: store [[TYPE]] 0, [[TYPE]]* [[PTR]], align 1

  ; 00
  ; EXIT-NEXT: [[OFFSET:%[0-9]+]] = add i64 [[SHADOW_BASE]], 115
  ; EXIT-NEXT: [[PTR:%[0-9]+]] = inttoptr i64 [[OFFSET]] to [[TYPE:i8]]*
  ; EXIT-NEXT: store [[TYPE]] 0, [[TYPE]]* [[PTR]], align 1

  ; 0000...
  ; EXIT-UAS-NEXT: [[OFFSET:%[0-9]+]] = add i64 [[SHADOW_BASE]], 0
  ; EXIT-UAS-NEXT: call void @__asan_set_shadow_00(i64 [[OFFSET]], i64 116)

  ; CHECK-NOT: add i64 [[SHADOW_BASE]]

  ret void
  ; CHECK: {{^[0-9]+}}:
  ; CHECK: ret void
}

declare void @foo(i32*)
define void @PR41481(i1 %b) sanitize_address {
; CHECK-LABEL: @PR41481
entry:
  %p1 = alloca i32
  %p2 = alloca i32
  %q1 = bitcast i32* %p1 to i8*
  %q2 = bitcast i32* %p2 to i8*
  br label %bb1

  ; Since we cannot account for all lifetime intrinsics in this function, we
  ; might have missed a lifetime.start one and therefore shouldn't poison the
  ; allocas at function entry.
  ; ENTRY: store i64 -935356719533264399
  ; ENTRY-UAS: store i64 -935356719533264399

bb1:
  %p = select i1 %b, i32* %p1, i32* %p2
  %q = select i1 %b, i8*  %q1, i8*  %q2
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %q)
  call void @foo(i32* %p)
  br i1 %b, label %bb2, label %bb3

bb2:
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %q1)
  br label %end

bb3:
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %q2)
  br label %end

end:
  ret void
}


declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture)
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture)

; CHECK-ON: declare void @__asan_set_shadow_00(i64, i64)
; CHECK-ON: declare void @__asan_set_shadow_f1(i64, i64)
; CHECK-ON: declare void @__asan_set_shadow_f2(i64, i64)
; CHECK-ON: declare void @__asan_set_shadow_f3(i64, i64)
; CHECK-ON: declare void @__asan_set_shadow_f5(i64, i64)
; CHECK-ON: declare void @__asan_set_shadow_f8(i64, i64)

; CHECK-OFF-NOT: declare void @__asan_set_shadow_
