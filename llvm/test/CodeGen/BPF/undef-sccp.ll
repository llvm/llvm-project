; RUN: opt -S -passes='sccp' %s -o %t1
; RUN: opt --bpf-check-undef-ir -S -mtriple=bpf-pc-linux %t1 >& %t2
; RUN: cat %t2 | FileCheck -check-prefixes=CHECK %s

%union.v6addr = type { %struct.anon.1 }
%struct.anon.1 = type { i64, i64 }
%union.macaddr = type { %struct.anon }
%struct.anon = type { i32, i16 }
%struct.icmp6hdr = type { i8, i8, i16, %union.anon }
%union.anon = type { [1 x i32] }
%struct.ipv6_opt_hdr = type { i8, i8 }

@repro.____fmt = internal constant [6 x i8] c"Start\00", align 1
@repro.____fmt.1 = internal constant [4 x i8] c"End\00", align 1
@__packed = dso_local global %union.v6addr zeroinitializer, align 8
@icmp6_ndisc_validate.____fmt = internal constant [23 x i8] c"pre ipv6_hdrlen_offset\00", align 1
@icmp6_ndisc_validate.____fmt.2 = internal constant [24 x i8] c"post ipv6_hdrlen_offset\00", align 1
@icmp6_ndisc_validate.____fmt.3 = internal constant [5 x i8] c"KO 1\00", align 1
@icmp6_ndisc_validate.____fmt.4 = internal constant [5 x i8] c"KO 2\00", align 1
@icmp6_ndisc_validate.____fmt.5 = internal constant [5 x i8] c"ACK \00", align 1
@ipv6_hdrlen_offset.____fmt = internal constant [17 x i8] c"OKOK %d, len: %d\00", align 1
@ipv6_hdrlen_offset.____fmt.6 = internal constant [18 x i8] c"KO INVALID EXTHDR\00", align 1
@llvm.compiler.used = appending global [1 x ptr] [ptr @repro], section "llvm.metadata"

define dso_local i32 @repro(ptr noundef %0) section "classifier" {
  %2 = alloca %struct.ipv6_opt_hdr, align 8
  %3 = tail call i64 (ptr, i32, ...) inttoptr (i64 6 to ptr)(ptr noundef nonnull @repro.____fmt, i32 noundef 6)
  %4 = tail call ptr asm sideeffect "$0 = *(u32 *)($1 + $2)", "=r,r,i"(ptr %0, i64 76)
  %5 = ptrtoint ptr %4 to i64
  %6 = trunc i64 %5 to i32
  %7 = tail call i64 (ptr, i32, ...) inttoptr (i64 6 to ptr)(ptr noundef nonnull @icmp6_ndisc_validate.____fmt, i32 noundef 23)
  call void @llvm.lifetime.start.p0(i64 2, ptr nonnull %2)
  %8 = getelementptr inbounds nuw i8, ptr %2, i64 1
  switch i8 undef, label %51 [
    i8 59, label %56
    i8 44, label %57
    i8 0, label %9
    i8 43, label %9
    i8 51, label %9
    i8 60, label %9
  ]

; CHECK:       unreachable in func repro, due to uninitialized variable?
; CHECK:       %8 = getelementptr inbounds nuw i8, ptr %2, i64 1
; CHECK-NEXT:  unreachable

9:                                                ; preds = %1, %1, %1, %1
  %10 = sub i32 40, %6
  %11 = call i64 inttoptr (i64 26 to ptr)(ptr noundef %0, i32 noundef %10, ptr noundef nonnull %2, i32 noundef 2)
  %12 = icmp slt i64 %11, 0
  br i1 %12, label %57, label %13

13:                                               ; preds = %9
  %14 = load i8, ptr %8, align 1
  %15 = zext i8 %14 to i32
  %16 = shl nuw nsw i32 %15, 3
  %17 = add nuw nsw i32 48, %16
  %18 = load i8, ptr %2, align 1
  switch i8 %18, label %51 [
    i8 59, label %56
    i8 44, label %57
    i8 0, label %19
    i8 43, label %19
    i8 51, label %19
    i8 60, label %19
  ]

19:                                               ; preds = %13, %13, %13, %13
  %20 = sub i32 %17, %6
  %21 = call i64 inttoptr (i64 26 to ptr)(ptr noundef %0, i32 noundef %20, ptr noundef nonnull %2, i32 noundef 2)
  %22 = icmp slt i64 %21, 0
  br i1 %22, label %57, label %23

23:                                               ; preds = %19
  %24 = icmp eq i8 %18, 51
  %25 = load i8, ptr %8, align 1
  %26 = zext i8 %25 to i32
  %27 = select i1 %24, i32 2, i32 3
  %28 = shl nuw nsw i32 %26, %27
  %29 = add nuw nsw i32 %17, 8
  %30 = add nuw nsw i32 %29, %28
  %31 = load i8, ptr %2, align 1
  switch i8 %31, label %51 [
    i8 59, label %56
    i8 44, label %57
    i8 0, label %32
    i8 43, label %32
    i8 51, label %32
    i8 60, label %32
  ]

32:                                               ; preds = %23, %23, %23, %23
  %33 = sub i32 %30, %6
  %34 = call i64 inttoptr (i64 26 to ptr)(ptr noundef %0, i32 noundef %33, ptr noundef nonnull %2, i32 noundef 2)
  %35 = icmp slt i64 %34, 0
  br i1 %35, label %57, label %36

36:                                               ; preds = %32
  %37 = icmp eq i8 %31, 51
  %38 = load i8, ptr %8, align 1
  %39 = zext i8 %38 to i32
  %40 = select i1 %37, i32 2, i32 3
  %41 = shl nuw nsw i32 %39, %40
  %42 = add nuw nsw i32 %30, 8
  %43 = add nuw nsw i32 %42, %41
  %44 = load i8, ptr %2, align 1
  switch i8 %44, label %51 [
    i8 59, label %56
    i8 44, label %57
    i8 0, label %45
    i8 43, label %45
    i8 51, label %45
    i8 60, label %45
  ]

45:                                               ; preds = %36, %36, %36, %36
  %46 = sub i32 %43, %6
  %47 = call i64 inttoptr (i64 26 to ptr)(ptr noundef %0, i32 noundef %46, ptr noundef nonnull %2, i32 noundef 2)
  %48 = icmp slt i64 %47, 0
  br i1 %48, label %57, label %49

49:                                               ; preds = %45
  %50 = call i64 (ptr, i32, ...) inttoptr (i64 6 to ptr)(ptr noundef nonnull @ipv6_hdrlen_offset.____fmt.6, i32 noundef 18)
  br label %59

51:                                               ; preds = %36, %23, %13, %1
  %52 = phi i8 [ undef, %1 ], [ %18, %13 ], [ %31, %23 ], [ %44, %36 ]
  %53 = phi i32 [ 40, %1 ], [ %17, %13 ], [ %30, %23 ], [ %43, %36 ]
  %54 = call i64 (ptr, i32, ...) inttoptr (i64 6 to ptr)(ptr noundef nonnull @ipv6_hdrlen_offset.____fmt, i32 noundef 17, i32 noundef 0, i32 noundef %53)
  %55 = icmp ne i8 %52, 58
  br label %59

56:                                               ; preds = %36, %23, %13, %1
  br label %59

57:                                               ; preds = %45, %36, %32, %23, %19, %13, %1, %9
  %58 = phi i32 [ -134, %9 ], [ -157, %1 ], [ -157, %13 ], [ -134, %19 ], [ -157, %23 ], [ -134, %32 ], [ -157, %36 ], [ -134, %45 ]
  br label %59

59:                                               ; preds = %57, %56, %51, %49
  %60 = phi i1 [ %55, %51 ], [ undef, %49 ], [ undef, %56 ], [ undef, %57 ]
  %61 = phi i32 [ %53, %51 ], [ -156, %49 ], [ -156, %56 ], [ %58, %57 ]
  call void @llvm.lifetime.end.p0(i64 2, ptr nonnull %2)
  %62 = call i64 (ptr, i32, ...) inttoptr (i64 6 to ptr)(ptr noundef nonnull @icmp6_ndisc_validate.____fmt.2, i32 noundef 24)
  %63 = icmp slt i32 %61, 0
  %64 = or i1 %63, %60
  br i1 %64, label %65, label %67

65:                                               ; preds = %59
  %66 = call i64 (ptr, i32, ...) inttoptr (i64 6 to ptr)(ptr noundef nonnull @icmp6_ndisc_validate.____fmt.3, i32 noundef 5)
  br label %77

67:                                               ; preds = %59
  %68 = call ptr asm sideeffect "$0 = *(u32 *)($1 + $2)", "=r,r,i"(ptr %0, i64 76)
  %69 = zext nneg i32 %61 to i64
  %70 = getelementptr inbounds nuw i8, ptr %68, i64 %69
  %71 = load i8, ptr %70, align 4
  %72 = icmp eq i8 %71, -121
  br i1 %72, label %75, label %73

73:                                               ; preds = %67
  %74 = call i64 (ptr, i32, ...) inttoptr (i64 6 to ptr)(ptr noundef nonnull @icmp6_ndisc_validate.____fmt.4, i32 noundef 5)
  br label %77

75:                                               ; preds = %67
  %76 = call i64 (ptr, i32, ...) inttoptr (i64 6 to ptr)(ptr noundef nonnull @icmp6_ndisc_validate.____fmt.5, i32 noundef 5)
  br label %77

77:                                               ; preds = %65, %73, %75
  %78 = call i64 (ptr, i32, ...) inttoptr (i64 6 to ptr)(ptr noundef nonnull @repro.____fmt.1, i32 noundef 4)
  ret i32 0
}
