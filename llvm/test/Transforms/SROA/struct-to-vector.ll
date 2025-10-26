; RUN: opt -passes='sroa,gvn,instcombine,simplifycfg' -S \
; RUN:   -sroa-max-struct-to-vector-bytes=16 %s \
; RUN:   | FileCheck %s \
; RUN:       --check-prefixes=FLAT,NESTED,PADDED,NONHOMO,I1,PTR
%struct.myint4 = type { i32, i32, i32, i32 }

; FLAT-LABEL: define dso_local void @foo_flat(
; FLAT-NOT: alloca
; FLAT-NOT: llvm.memcpy
; FLAT-NOT: llvm.memset
; FLAT: insertelement <2 x i64>
; FLAT: bitcast <2 x i64> %{{[^ ]+}} to <4 x i32>
; FLAT: select i1 %{{[^,]+}}, <4 x i32> zeroinitializer, <4 x i32> %{{[^)]+}}
; FLAT: store <4 x i32> %{{[^,]+}}, ptr %x, align 16
; FLAT: ret void
define dso_local void @foo_flat(ptr noundef %x, i64 %y.coerce0, i64 %y.coerce1, i32 noundef %cond) {
entry:
  %y = alloca %struct.myint4, align 16
  %x.addr = alloca ptr, align 8
  %cond.addr = alloca i32, align 4
  %temp = alloca %struct.myint4, align 16
  %zero = alloca %struct.myint4, align 16
  %data = alloca %struct.myint4, align 16
  %0 = getelementptr inbounds nuw { i64, i64 }, ptr %y, i32 0, i32 0
  store i64 %y.coerce0, ptr %0, align 16
  %1 = getelementptr inbounds nuw { i64, i64 }, ptr %y, i32 0, i32 1
  store i64 %y.coerce1, ptr %1, align 8
  store ptr %x, ptr %x.addr, align 8
  store i32 %cond, ptr %cond.addr, align 4
  call void @llvm.lifetime.start.p0(ptr %temp)
  call void @llvm.memcpy.p0.p0.i64(ptr align 16 %temp, ptr align 16 %y, i64 16, i1 false)
  call void @llvm.lifetime.start.p0(ptr %zero)
  call void @llvm.memset.p0.i64(ptr align 16 %zero, i8 0, i64 16, i1 false)
  call void @llvm.lifetime.start.p0(ptr %data)
  %2 = load i32, ptr %cond.addr, align 4
  %tobool = icmp ne i32 %2, 0
  br i1 %tobool, label %cond.true, label %cond.false

cond.true:
  br label %cond.end

cond.false:
  br label %cond.end

cond.end:
  %cond1 = phi ptr [ %temp, %cond.true ], [ %zero, %cond.false ]
  call void @llvm.memcpy.p0.p0.i64(ptr align 16 %data, ptr align 16 %cond1, i64 16, i1 false)
  %3 = load ptr, ptr %x.addr, align 8
  call void @llvm.memcpy.p0.p0.i64(ptr align 16 %3, ptr align 16 %data, i64 16, i1 false)
  call void @llvm.lifetime.end.p0(ptr %data)
  call void @llvm.lifetime.end.p0(ptr %zero)
  call void @llvm.lifetime.end.p0(ptr %temp)
  ret void
}
%struct.myint4_base_n = type { i32, i32, i32, i32 }
%struct.myint4_nested = type { %struct.myint4_base_n }

; NESTED-LABEL: define dso_local void @foo_nested(
; NESTED-NOT: alloca
; NESTED-NOT: llvm.memcpy
; NESTED-NOT: llvm.memset
; NESTED: insertelement <2 x i64>
; NESTED: bitcast <2 x i64> %{{[^ ]+}} to <4 x i32>
; NESTED: select i1 %{{[^,]+}}, <4 x i32> zeroinitializer, <4 x i32> %{{[^)]+}}
; NESTED: store <4 x i32> %{{[^,]+}}, ptr %x, align 16
; NESTED: ret void
define dso_local void @foo_nested(ptr noundef %x, i64 %y.coerce0, i64 %y.coerce1, i32 noundef %cond) {
entry:
  %y = alloca %struct.myint4_nested, align 16
  %x.addr = alloca ptr, align 8
  %cond.addr = alloca i32, align 4
  %temp = alloca %struct.myint4_nested, align 16
  %zero = alloca %struct.myint4_nested, align 16
  %data = alloca %struct.myint4_nested, align 16
  %0 = getelementptr inbounds nuw { i64, i64 }, ptr %y, i32 0, i32 0
  store i64 %y.coerce0, ptr %0, align 16
  %1 = getelementptr inbounds nuw { i64, i64 }, ptr %y, i32 0, i32 1
  store i64 %y.coerce1, ptr %1, align 8
  store ptr %x, ptr %x.addr, align 8
  store i32 %cond, ptr %cond.addr, align 4
  call void @llvm.lifetime.start.p0(ptr %temp)
  call void @llvm.memcpy.p0.p0.i64(ptr align 16 %temp, ptr align 16 %y, i64 16, i1 false)
  call void @llvm.lifetime.start.p0(ptr %zero)
  call void @llvm.memset.p0.i64(ptr align 16 %zero, i8 0, i64 16, i1 false)
  call void @llvm.lifetime.start.p0(ptr %data)
  %2 = load i32, ptr %cond.addr, align 4
  %tobool = icmp ne i32 %2, 0
  br i1 %tobool, label %cond.true, label %cond.false

cond.true:
  br label %cond.end

cond.false:
  br label %cond.end

cond.end:
  %cond1 = phi ptr [ %temp, %cond.true ], [ %zero, %cond.false ]
  call void @llvm.memcpy.p0.p0.i64(ptr align 16 %data, ptr align 16 %cond1, i64 16, i1 false)
  %3 = load ptr, ptr %x.addr, align 8
  call void @llvm.memcpy.p0.p0.i64(ptr align 16 %3, ptr align 16 %data, i64 16, i1 false)
  call void @llvm.lifetime.end.p0(ptr %data)
  call void @llvm.lifetime.end.p0(ptr %zero)
  call void @llvm.lifetime.end.p0(ptr %temp)
  ret void
}

; PADDED-LABEL: define dso_local void @foo_padded(
; PADDED: llvm.memcpy
; PADDED-NOT: store <
; PADDED: ret void
%struct.padded = type { i32, i8, i32, i8 }
define dso_local void @foo_padded(ptr noundef %x, i32 %a0, i8 %a1,
                                  i32 %a2, i8 %a3,
                                  i32 noundef %cond) {
entry:
  %y = alloca %struct.padded, align 4
  %x.addr = alloca ptr, align 8
  %cond.addr = alloca i32, align 4
  %temp = alloca %struct.padded, align 4
  %zero = alloca %struct.padded, align 4
  %data = alloca %struct.padded, align 4
  %y_i32_0 = getelementptr inbounds %struct.padded, ptr %y, i32 0, i32 0
  store i32 %a0, ptr %y_i32_0, align 4
  %y_i8_1 = getelementptr inbounds %struct.padded, ptr %y, i32 0, i32 1
  store i8 %a1, ptr %y_i8_1, align 1
  %y_i32_2 = getelementptr inbounds %struct.padded, ptr %y, i32 0, i32 2
  store i32 %a2, ptr %y_i32_2, align 4
  %y_i8_3 = getelementptr inbounds %struct.padded, ptr %y, i32 0, i32 3
  store i8 %a3, ptr %y_i8_3, align 1
  store ptr %x, ptr %x.addr, align 8
  store i32 %cond, ptr %cond.addr, align 4
  call void @llvm.lifetime.start.p0(ptr %temp)
  call void @llvm.memcpy.p0.p0.i64(ptr align 4 %temp, ptr align 4 %y,
                                   i64 16, i1 false)
  call void @llvm.lifetime.start.p0(ptr %zero)
  call void @llvm.memset.p0.i64(ptr align 4 %zero, i8 0, i64 16, i1 false)
  call void @llvm.lifetime.start.p0(ptr %data)
  %c.pad = load i32, ptr %cond.addr, align 4
  %tobool.pad = icmp ne i32 %c.pad, 0
  br i1 %tobool.pad, label %cond.true.pad, label %cond.false.pad

cond.true.pad:
  br label %cond.end.pad

cond.false.pad:
  br label %cond.end.pad

cond.end.pad:
  %cond1.pad = phi ptr [ %temp, %cond.true.pad ], [ %zero, %cond.false.pad ]
  call void @llvm.memcpy.p0.p0.i64(ptr align 4 %data, ptr align 4 %cond1.pad,
                                   i64 16, i1 false)
  %xv.pad = load ptr, ptr %x.addr, align 8
  call void @llvm.memcpy.p0.p0.i64(ptr align 4 %xv.pad, ptr align 4 %data,
                                   i64 16, i1 false)
  call void @llvm.lifetime.end.p0(ptr %data)
  call void @llvm.lifetime.end.p0(ptr %zero)
  call void @llvm.lifetime.end.p0(ptr %temp)
  ret void
}

; NONHOMO-LABEL: define dso_local void @foo_nonhomo(
; NONHOMO: llvm.memcpy
; NONHOMO-NOT: store <
; NONHOMO: ret void
%struct.nonhomo = type { i32, i64, i32, i64 }
define dso_local void @foo_nonhomo(ptr noundef %x, i32 %a0, i64 %a1,
                                   i32 %a2, i64 %a3,
                                   i32 noundef %cond) {
entry:
  %y = alloca %struct.nonhomo, align 8
  %x.addr = alloca ptr, align 8
  %cond.addr = alloca i32, align 4
  %temp = alloca %struct.nonhomo, align 8
  %zero = alloca %struct.nonhomo, align 8
  %data = alloca %struct.nonhomo, align 8
  %y_i32_0n = getelementptr inbounds %struct.nonhomo, ptr %y, i32 0, i32 0
  store i32 %a0, ptr %y_i32_0n, align 4
  %y_i64_1n = getelementptr inbounds %struct.nonhomo, ptr %y, i32 0, i32 1
  store i64 %a1, ptr %y_i64_1n, align 8
  %y_i32_2n = getelementptr inbounds %struct.nonhomo, ptr %y, i32 0, i32 2
  store i32 %a2, ptr %y_i32_2n, align 4
  %y_i64_3n = getelementptr inbounds %struct.nonhomo, ptr %y, i32 0, i32 3
  store i64 %a3, ptr %y_i64_3n, align 8
  store ptr %x, ptr %x.addr, align 8
  store i32 %cond, ptr %cond.addr, align 4
  call void @llvm.lifetime.start.p0(ptr %temp)
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %temp, ptr align 8 %y,
                                   i64 32, i1 false)
  call void @llvm.lifetime.start.p0(ptr %zero)
  call void @llvm.memset.p0.i64(ptr align 8 %zero, i8 0, i64 32, i1 false)
  call void @llvm.lifetime.start.p0(ptr %data)
  %c.nh = load i32, ptr %cond.addr, align 4
  %tobool.nh = icmp ne i32 %c.nh, 0
  br i1 %tobool.nh, label %cond.true.nh, label %cond.false.nh

cond.true.nh:
  br label %cond.end.nh

cond.false.nh:
  br label %cond.end.nh

cond.end.nh:
  %cond1.nh = phi ptr [ %temp, %cond.true.nh ], [ %zero, %cond.false.nh ]
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %data, ptr align 8 %cond1.nh,
                                   i64 32, i1 false)
  %xv.nh = load ptr, ptr %x.addr, align 8
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %xv.nh, ptr align 8 %data,
                                   i64 32, i1 false)
  call void @llvm.lifetime.end.p0(ptr %data)
  call void @llvm.lifetime.end.p0(ptr %zero)
  call void @llvm.lifetime.end.p0(ptr %temp)
  ret void
}

; I1-LABEL: define dso_local void @foo_i1(
; I1-NOT: <4 x i1>
; I1: ret void
%struct.i1x4 = type { i1, i1, i1, i1 }
define dso_local void @foo_i1(ptr noundef %x, i64 %dummy0, i64 %dummy1,
                              i32 noundef %cond) {
entry:
  %y = alloca %struct.i1x4, align 1
  %x.addr = alloca ptr, align 8
  %cond.addr = alloca i32, align 4
  %temp = alloca %struct.i1x4, align 1
  %zero = alloca %struct.i1x4, align 1
  %data = alloca %struct.i1x4, align 1
  store ptr %x, ptr %x.addr, align 8
  store i32 %cond, ptr %cond.addr, align 4
  call void @llvm.lifetime.start.p0(ptr %temp)
  call void @llvm.memcpy.p0.p0.i64(ptr align 1 %temp, ptr align 1 %y,
                                   i64 4, i1 false)
  call void @llvm.lifetime.start.p0(ptr %zero)
  call void @llvm.memset.p0.i64(ptr align 1 %zero, i8 0, i64 4, i1 false)
  call void @llvm.lifetime.start.p0(ptr %data)
  %c.i1 = load i32, ptr %cond.addr, align 4
  %tobool.i1 = icmp ne i32 %c.i1, 0
  br i1 %tobool.i1, label %cond.true.i1, label %cond.false.i1

cond.true.i1:
  br label %cond.end.i1

cond.false.i1:
  br label %cond.end.i1

cond.end.i1:
  %cond1.i1 = phi ptr [ %temp, %cond.true.i1 ], [ %zero, %cond.false.i1 ]
  call void @llvm.memcpy.p0.p0.i64(ptr align 1 %data, ptr align 1 %cond1.i1,
                                   i64 4, i1 false)
  %xv.i1 = load ptr, ptr %x.addr, align 8
  call void @llvm.memcpy.p0.p0.i64(ptr align 1 %xv.i1, ptr align 1 %data,
                                   i64 4, i1 false)
  call void @llvm.lifetime.end.p0(ptr %data)
  call void @llvm.lifetime.end.p0(ptr %zero)
  call void @llvm.lifetime.end.p0(ptr %temp)
  ret void
}

; PTR-LABEL: define dso_local void @foo_ptr(
; PTR: llvm.memcpy
; PTR-NOT: <4 x ptr>
; PTR: ret void
%struct.ptr4 = type { ptr, ptr, ptr, ptr }
define dso_local void @foo_ptr(ptr noundef %x, ptr %p0, ptr %p1,
                               ptr %p2, ptr %p3,
                               i32 noundef %cond) {
entry:
  %y = alloca %struct.ptr4, align 8
  %x.addr = alloca ptr, align 8
  %cond.addr = alloca i32, align 4
  %temp = alloca %struct.ptr4, align 8
  %zero = alloca %struct.ptr4, align 8
  %data = alloca %struct.ptr4, align 8
  %y_p0 = getelementptr inbounds %struct.ptr4, ptr %y, i32 0, i32 0
  store ptr %p0, ptr %y_p0, align 8
  %y_p1 = getelementptr inbounds %struct.ptr4, ptr %y, i32 0, i32 1
  store ptr %p1, ptr %y_p1, align 8
  %y_p2 = getelementptr inbounds %struct.ptr4, ptr %y, i32 0, i32 2
  store ptr %p2, ptr %y_p2, align 8
  %y_p3 = getelementptr inbounds %struct.ptr4, ptr %y, i32 0, i32 3
  store ptr %p3, ptr %y_p3, align 8
  store ptr %x, ptr %x.addr, align 8
  store i32 %cond, ptr %cond.addr, align 4
  call void @llvm.lifetime.start.p0(ptr %temp)
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %temp, ptr align 8 %y,
                                   i64 32, i1 false)
  call void @llvm.lifetime.start.p0(ptr %zero)
  call void @llvm.memset.p0.i64(ptr align 8 %zero, i8 0, i64 32, i1 false)
  call void @llvm.lifetime.start.p0(ptr %data)
  %c.ptr = load i32, ptr %cond.addr, align 4
  %tobool.ptr = icmp ne i32 %c.ptr, 0
  br i1 %tobool.ptr, label %cond.true.ptr, label %cond.false.ptr

cond.true.ptr:
  br label %cond.end.ptr

cond.false.ptr:
  br label %cond.end.ptr

cond.end.ptr:
  %cond1.ptr = phi ptr [ %temp, %cond.true.ptr ], [ %zero, %cond.false.ptr ]
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %data, ptr align 8 %cond1.ptr,
                                   i64 32, i1 false)
  %xv.ptr = load ptr, ptr %x.addr, align 8
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %xv.ptr, ptr align 8 %data,
                                   i64 32, i1 false)
  call void @llvm.lifetime.end.p0(ptr %data)
  call void @llvm.lifetime.end.p0(ptr %zero)
  call void @llvm.lifetime.end.p0(ptr %temp)
  ret void
}
