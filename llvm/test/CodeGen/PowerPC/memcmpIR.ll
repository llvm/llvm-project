; RUN: llc -o - -mtriple=powerpc64le-unknown-gnu-linux -stop-after codegenprepare %s | FileCheck %s
; RUN: llc -o - -mtriple=powerpc64-unknown-gnu-linux -stop-after codegenprepare %s | FileCheck %s --check-prefix=CHECK-BE

define signext i32 @test1(ptr nocapture readonly %buffer1, ptr nocapture readonly %buffer2)  {
entry:
  ; CHECK-LABEL: @test1(
  ; CHECK-LABEL: res_block:{{.*}}
  ; CHECK: [[ICMP2:%[0-9]+]] = icmp ult i64
  ; CHECK-NEXT: [[SELECT:%[0-9]+]] = select i1 [[ICMP2]], i32 -1, i32 1
  ; CHECK-NEXT: br label %endblock

  ; CHECK-LABEL: loadbb:{{.*}}
  ; CHECK: [[LOAD1:%[0-9]+]] = load i64, ptr
  ; CHECK-NEXT: [[LOAD2:%[0-9]+]] = load i64, ptr
  ; CHECK-NEXT: [[BSWAP1:%[0-9]+]] = call i64 @llvm.bswap.i64(i64 [[LOAD1]])
  ; CHECK-NEXT: [[BSWAP2:%[0-9]+]] = call i64 @llvm.bswap.i64(i64 [[LOAD2]])
  ; CHECK-NEXT: [[ICMP:%[0-9]+]] = icmp eq i64 [[BSWAP1]], [[BSWAP2]]
  ; CHECK-NEXT:  br i1 [[ICMP]], label %loadbb1, label %res_block

  ; CHECK-LABEL: loadbb1:{{.*}}
  ; CHECK-NEXT: [[GEP1:%[0-9]+]] = getelementptr i8, ptr {{.*}}, i64 8
  ; CHECK-NEXT: [[GEP2:%[0-9]+]] = getelementptr i8, ptr {{.*}}, i64 8
  ; CHECK-NEXT: [[LOAD1:%[0-9]+]] = load i64, ptr [[GEP1]]
  ; CHECK-NEXT: [[LOAD2:%[0-9]+]] = load i64, ptr [[GEP2]]
  ; CHECK-NEXT: [[BSWAP1:%[0-9]+]] = call i64 @llvm.bswap.i64(i64 [[LOAD1]])
  ; CHECK-NEXT: [[BSWAP2:%[0-9]+]] = call i64 @llvm.bswap.i64(i64 [[LOAD2]])
  ; CHECK-NEXT: [[ICMP:%[0-9]+]] = icmp eq i64 [[BSWAP1]], [[BSWAP2]]
  ; CHECK-NEXT:  br i1 [[ICMP]], label %endblock, label %res_block

  ; CHECK-BE-LABEL: @test1(
  ; CHECK-BE-LABEL: res_block:{{.*}}
  ; CHECK-BE: [[ICMP2:%[0-9]+]] = icmp ult i64
  ; CHECK-BE-NEXT: [[SELECT:%[0-9]+]] = select i1 [[ICMP2]], i32 -1, i32 1
  ; CHECK-BE-NEXT: br label %endblock

  ; CHECK-BE-LABEL: loadbb:{{.*}}
  ; CHECK-BE: [[LOAD1:%[0-9]+]] = load i64, ptr
  ; CHECK-BE-NEXT: [[LOAD2:%[0-9]+]] = load i64, ptr
  ; CHECK-BE-NEXT: [[ICMP:%[0-9]+]] = icmp eq i64 [[LOAD1]], [[LOAD2]]
  ; CHECK-BE-NEXT:  br i1 [[ICMP]], label %loadbb1, label %res_block

  ; CHECK-BE-LABEL: loadbb1:{{.*}}
  ; CHECK-BE-NEXT: [[GEP1:%[0-9]+]] = getelementptr i8, ptr {{.*}}, i64 8
  ; CHECK-BE-NEXT: [[GEP2:%[0-9]+]] = getelementptr i8, ptr {{.*}}, i64 8
  ; CHECK-BE-NEXT: [[LOAD1:%[0-9]+]] = load i64, ptr [[GEP1]]
  ; CHECK-BE-NEXT: [[LOAD2:%[0-9]+]] = load i64, ptr [[GEP2]]
  ; CHECK-BE-NEXT: [[ICMP:%[0-9]+]] = icmp eq i64 [[LOAD1]], [[LOAD2]]
  ; CHECK-BE-NEXT:  br i1 [[ICMP]], label %endblock, label %res_block

  %call = tail call signext i32 @memcmp(ptr %buffer1, ptr %buffer2, i64 16)
  ret i32 %call
}

declare signext i32 @memcmp(ptr nocapture, ptr nocapture, i64) local_unnamed_addr #1

define signext i32 @test2(ptr nocapture readonly %buffer1, ptr nocapture readonly %buffer2)  {
  ; CHECK-LABEL: @test2(
  ; CHECK: [[LOAD1:%[0-9]+]] = load i32, ptr
  ; CHECK-NEXT: [[LOAD2:%[0-9]+]] = load i32, ptr
  ; CHECK-NEXT: [[BSWAP1:%[0-9]+]] = call i32 @llvm.bswap.i32(i32 [[LOAD1]])
  ; CHECK-NEXT: [[BSWAP2:%[0-9]+]] = call i32 @llvm.bswap.i32(i32 [[LOAD2]])
  ; CHECK-NEXT: [[UCMP:%[0-9]+]] = call i32 @llvm.ucmp.i32.i32(i32 [[BSWAP1]], i32 [[BSWAP2]])
  ; CHECK-NEXT: ret i32 [[UCMP]]

  ; CHECK-BE-LABEL: @test2(
  ; CHECK-BE: [[LOAD1:%[0-9]+]] = load i32, ptr
  ; CHECK-BE-NEXT: [[LOAD2:%[0-9]+]] = load i32, ptr
  ; CHECK-BE-NEXT: [[UCMP:%[0-9]+]] = call i32 @llvm.ucmp.i32.i32(i32 [[LOAD1]], i32 [[LOAD2]])
  ; CHECK-BE-NEXT: ret i32 [[UCMP]]

entry:
  %call = tail call signext i32 @memcmp(ptr %buffer1, ptr %buffer2, i64 4)
  ret i32 %call
}

define signext i32 @test3(ptr nocapture readonly %buffer1, ptr nocapture readonly %buffer2)  {
  ; CHECK-LABEL: res_block:{{.*}}
  ; CHECK: [[ICMP2:%[0-9]+]] = icmp ult i64
  ; CHECK-NEXT: [[SELECT:%[0-9]+]] = select i1 [[ICMP2]], i32 -1, i32 1
  ; CHECK-NEXT: br label %endblock

  ; CHECK-LABEL: loadbb:{{.*}}
  ; CHECK: [[LOAD1:%[0-9]+]] = load i64, ptr
  ; CHECK-NEXT: [[LOAD2:%[0-9]+]] = load i64, ptr
  ; CHECK-NEXT: [[BSWAP1:%[0-9]+]] = call i64 @llvm.bswap.i64(i64 [[LOAD1]])
  ; CHECK-NEXT: [[BSWAP2:%[0-9]+]] = call i64 @llvm.bswap.i64(i64 [[LOAD2]])
  ; CHECK-NEXT: [[ICMP:%[0-9]+]] = icmp eq i64 [[BSWAP1]], [[BSWAP2]]
  ; CHECK-NEXT:  br i1 [[ICMP]], label %loadbb1, label %res_block

  ; CHECK-LABEL: loadbb1:{{.*}}
  ; CHECK: [[LOAD1:%[0-9]+]] = load i32, ptr
  ; CHECK-NEXT: [[LOAD2:%[0-9]+]] = load i32, ptr
  ; CHECK-NEXT: [[BSWAP1:%[0-9]+]] = call i32 @llvm.bswap.i32(i32 [[LOAD1]])
  ; CHECK-NEXT: [[BSWAP2:%[0-9]+]] = call i32 @llvm.bswap.i32(i32 [[LOAD2]])
  ; CHECK-NEXT: [[ZEXT1:%[0-9]+]] = zext i32 [[BSWAP1]] to i64
  ; CHECK-NEXT: [[ZEXT2:%[0-9]+]] = zext i32 [[BSWAP2]] to i64
  ; CHECK-NEXT: [[ICMP:%[0-9]+]] = icmp eq i64 [[ZEXT1]], [[ZEXT2]]
  ; CHECK-NEXT:  br i1 [[ICMP]], label %loadbb2, label %res_block

  ; CHECK-LABEL: loadbb2:{{.*}}
  ; CHECK: [[LOAD1:%[0-9]+]] = load i16, ptr
  ; CHECK-NEXT: [[LOAD2:%[0-9]+]] = load i16, ptr
  ; CHECK-NEXT: [[BSWAP1:%[0-9]+]] = call i16 @llvm.bswap.i16(i16 [[LOAD1]])
  ; CHECK-NEXT: [[BSWAP2:%[0-9]+]] = call i16 @llvm.bswap.i16(i16 [[LOAD2]])
  ; CHECK-NEXT: [[ZEXT1:%[0-9]+]] = zext i16 [[BSWAP1]] to i64
  ; CHECK-NEXT: [[ZEXT2:%[0-9]+]] = zext i16 [[BSWAP2]] to i64
  ; CHECK-NEXT: [[ICMP:%[0-9]+]] = icmp eq i64 [[ZEXT1]], [[ZEXT2]]
  ; CHECK-NEXT:  br i1 [[ICMP]], label %loadbb3, label %res_block

  ; CHECK-LABEL: loadbb3:{{.*}}
  ; CHECK: [[LOAD1:%[0-9]+]] = load i8, ptr
  ; CHECK-NEXT: [[LOAD2:%[0-9]+]] = load i8, ptr
  ; CHECK-NEXT: [[ZEXT1:%[0-9]+]] = zext i8 [[LOAD1]] to i32
  ; CHECK-NEXT: [[ZEXT2:%[0-9]+]] = zext i8 [[LOAD2]] to i32
  ; CHECK-NEXT: [[SUB:%[0-9]+]] = sub i32 [[ZEXT1]], [[ZEXT2]]
  ; CHECK-NEXT:  br label %endblock

  ; CHECK-BE-LABEL: res_block:{{.*}}
  ; CHECK-BE: [[ICMP2:%[0-9]+]] = icmp ult i64
  ; CHECK-BE-NEXT: [[SELECT:%[0-9]+]] = select i1 [[ICMP2]], i32 -1, i32 1
  ; CHECK-BE-NEXT: br label %endblock

  ; CHECK-BE-LABEL: loadbb:{{.*}}
  ; CHECK-BE: [[LOAD1:%[0-9]+]] = load i64, ptr
  ; CHECK-BE-NEXT: [[LOAD2:%[0-9]+]] = load i64, ptr
  ; CHECK-BE-NEXT: [[ICMP:%[0-9]+]] = icmp eq i64 [[LOAD1]], [[LOAD2]]
  ; CHECK-BE-NEXT:  br i1 [[ICMP]], label %loadbb1, label %res_block

  ; CHECK-BE: [[LOAD1:%[0-9]+]] = load i32, ptr
  ; CHECK-BE-NEXT: [[LOAD2:%[0-9]+]] = load i32, ptr
  ; CHECK-BE-NEXT: [[ZEXT1:%[0-9]+]] = zext i32 [[LOAD1]] to i64
  ; CHECK-BE-NEXT: [[ZEXT2:%[0-9]+]] = zext i32 [[LOAD2]] to i64
  ; CHECK-BE-NEXT: [[ICMP:%[0-9]+]] = icmp eq i64 [[ZEXT1]], [[ZEXT2]]
  ; CHECK-BE-NEXT:  br i1 [[ICMP]], label %loadbb2, label %res_block

  ; CHECK-BE: [[LOAD1:%[0-9]+]] = load i16, ptr
  ; CHECK-BE-NEXT: [[LOAD2:%[0-9]+]] = load i16, ptr
  ; CHECK-BE-NEXT: [[ZEXT1:%[0-9]+]] = zext i16 [[LOAD1]] to i64
  ; CHECK-BE-NEXT: [[ZEXT2:%[0-9]+]] = zext i16 [[LOAD2]] to i64
  ; CHECK-BE-NEXT: [[ICMP:%[0-9]+]] = icmp eq i64 [[ZEXT1]], [[ZEXT2]]
  ; CHECK-BE-NEXT:  br i1 [[ICMP]], label %loadbb3, label %res_block

  ; CHECK-BE: [[LOAD1:%[0-9]+]] = load i8, ptr
  ; CHECK-BE-NEXT: [[LOAD2:%[0-9]+]] = load i8, ptr
  ; CHECK-BE-NEXT: [[ZEXT1:%[0-9]+]] = zext i8 [[LOAD1]] to i32
  ; CHECK-BE-NEXT: [[ZEXT2:%[0-9]+]] = zext i8 [[LOAD2]] to i32
  ; CHECK-BE-NEXT: [[SUB:%[0-9]+]] = sub i32 [[ZEXT1]], [[ZEXT2]]
  ; CHECK-BE-NEXT:  br label %endblock

entry:
  %call = tail call signext i32 @memcmp(ptr %buffer1, ptr %buffer2, i64 15)
  ret i32 %call
}
  ; CHECK: call = tail call signext i32 @memcmp
  ; CHECK-BE: call = tail call signext i32 @memcmp
define signext i32 @test4(ptr nocapture readonly %buffer1, ptr nocapture readonly %buffer2)  {

entry:
  %call = tail call signext i32 @memcmp(ptr %buffer1, ptr %buffer2, i64 65)
  ret i32 %call
}

define signext i32 @test5(ptr nocapture readonly %buffer1, ptr nocapture readonly %buffer2, i32 signext %SIZE)  {
  ; CHECK: call = tail call signext i32 @memcmp
  ; CHECK-BE: call = tail call signext i32 @memcmp
entry:
  %conv = sext i32 %SIZE to i64
  %call = tail call signext i32 @memcmp(ptr %buffer1, ptr %buffer2, i64 %conv)
  ret i32 %call
}
