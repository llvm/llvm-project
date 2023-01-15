; RUN: llc  < %s -march=mipsel | FileCheck %s

@a0 = external global i32
@b0 = external global ptr
@a1 = external global i32
@b1 = external global ptr
@a2 = external global i32
@b2 = external global ptr
@a3 = external global i32
@b3 = external global ptr
@a4 = external global i32
@b4 = external global ptr
@a5 = external global i32
@b5 = external global ptr
@a6 = external global i32
@b6 = external global ptr
@a7 = external global i32
@b7 = external global ptr
@a8 = external global i32
@b8 = external global ptr
@a9 = external global i32
@b9 = external global ptr
@a10 = external global i32
@b10 = external global ptr
@a11 = external global i32
@b11 = external global ptr
@a12 = external global i32
@b12 = external global ptr
@a13 = external global i32
@b13 = external global ptr
@a14 = external global i32
@b14 = external global ptr
@a15 = external global i32
@b15 = external global ptr
@a16 = external global i32
@b16 = external global ptr
@a17 = external global i32
@b17 = external global ptr
@a18 = external global i32
@b18 = external global ptr
@a19 = external global i32
@b19 = external global ptr
@a20 = external global i32
@b20 = external global ptr
@a21 = external global i32
@b21 = external global ptr
@a22 = external global i32
@b22 = external global ptr
@a23 = external global i32
@b23 = external global ptr
@a24 = external global i32
@b24 = external global ptr
@a25 = external global i32
@b25 = external global ptr
@a26 = external global i32
@b26 = external global ptr
@a27 = external global i32
@b27 = external global ptr
@a28 = external global i32
@b28 = external global ptr
@a29 = external global i32
@b29 = external global ptr
@c0 = external global ptr
@c1 = external global ptr
@c2 = external global ptr
@c3 = external global ptr
@c4 = external global ptr
@c5 = external global ptr
@c6 = external global ptr
@c7 = external global ptr
@c8 = external global ptr
@c9 = external global ptr
@c10 = external global ptr
@c11 = external global ptr
@c12 = external global ptr
@c13 = external global ptr
@c14 = external global ptr
@c15 = external global ptr
@c16 = external global ptr
@c17 = external global ptr
@c18 = external global ptr
@c19 = external global ptr
@c20 = external global ptr
@c21 = external global ptr
@c22 = external global ptr
@c23 = external global ptr
@c24 = external global ptr
@c25 = external global ptr
@c26 = external global ptr
@c27 = external global ptr
@c28 = external global ptr
@c29 = external global ptr

define i32 @f1() nounwind {
entry:
; CHECK: sw  $ra, {{[0-9]+}}($sp)            # 4-byte Folded Spill
; CHECK: $ra
; CHECK: lw  $ra, {{[0-9]+}}($sp)            # 4-byte Folded Reload
; CHECK: jr  $ra

  %0 = load i32, ptr @a0, align 4
  %1 = load ptr, ptr @b0, align 4
  store i32 %0, ptr %1, align 4
  %2 = load i32, ptr @a1, align 4
  %3 = load ptr, ptr @b1, align 4
  store i32 %2, ptr %3, align 4
  %4 = load i32, ptr @a2, align 4
  %5 = load ptr, ptr @b2, align 4
  store i32 %4, ptr %5, align 4
  %6 = load i32, ptr @a3, align 4
  %7 = load ptr, ptr @b3, align 4
  store i32 %6, ptr %7, align 4
  %8 = load i32, ptr @a4, align 4
  %9 = load ptr, ptr @b4, align 4
  store i32 %8, ptr %9, align 4
  %10 = load i32, ptr @a5, align 4
  %11 = load ptr, ptr @b5, align 4
  store i32 %10, ptr %11, align 4
  %12 = load i32, ptr @a6, align 4
  %13 = load ptr, ptr @b6, align 4
  store i32 %12, ptr %13, align 4
  %14 = load i32, ptr @a7, align 4
  %15 = load ptr, ptr @b7, align 4
  store i32 %14, ptr %15, align 4
  %16 = load i32, ptr @a8, align 4
  %17 = load ptr, ptr @b8, align 4
  store i32 %16, ptr %17, align 4
  %18 = load i32, ptr @a9, align 4
  %19 = load ptr, ptr @b9, align 4
  store i32 %18, ptr %19, align 4
  %20 = load i32, ptr @a10, align 4
  %21 = load ptr, ptr @b10, align 4
  store i32 %20, ptr %21, align 4
  %22 = load i32, ptr @a11, align 4
  %23 = load ptr, ptr @b11, align 4
  store i32 %22, ptr %23, align 4
  %24 = load i32, ptr @a12, align 4
  %25 = load ptr, ptr @b12, align 4
  store i32 %24, ptr %25, align 4
  %26 = load i32, ptr @a13, align 4
  %27 = load ptr, ptr @b13, align 4
  store i32 %26, ptr %27, align 4
  %28 = load i32, ptr @a14, align 4
  %29 = load ptr, ptr @b14, align 4
  store i32 %28, ptr %29, align 4
  %30 = load i32, ptr @a15, align 4
  %31 = load ptr, ptr @b15, align 4
  store i32 %30, ptr %31, align 4
  %32 = load i32, ptr @a16, align 4
  %33 = load ptr, ptr @b16, align 4
  store i32 %32, ptr %33, align 4
  %34 = load i32, ptr @a17, align 4
  %35 = load ptr, ptr @b17, align 4
  store i32 %34, ptr %35, align 4
  %36 = load i32, ptr @a18, align 4
  %37 = load ptr, ptr @b18, align 4
  store i32 %36, ptr %37, align 4
  %38 = load i32, ptr @a19, align 4
  %39 = load ptr, ptr @b19, align 4
  store i32 %38, ptr %39, align 4
  %40 = load i32, ptr @a20, align 4
  %41 = load ptr, ptr @b20, align 4
  store i32 %40, ptr %41, align 4
  %42 = load i32, ptr @a21, align 4
  %43 = load ptr, ptr @b21, align 4
  store i32 %42, ptr %43, align 4
  %44 = load i32, ptr @a22, align 4
  %45 = load ptr, ptr @b22, align 4
  store i32 %44, ptr %45, align 4
  %46 = load i32, ptr @a23, align 4
  %47 = load ptr, ptr @b23, align 4
  store i32 %46, ptr %47, align 4
  %48 = load i32, ptr @a24, align 4
  %49 = load ptr, ptr @b24, align 4
  store i32 %48, ptr %49, align 4
  %50 = load i32, ptr @a25, align 4
  %51 = load ptr, ptr @b25, align 4
  store i32 %50, ptr %51, align 4
  %52 = load i32, ptr @a26, align 4
  %53 = load ptr, ptr @b26, align 4
  store i32 %52, ptr %53, align 4
  %54 = load i32, ptr @a27, align 4
  %55 = load ptr, ptr @b27, align 4
  store i32 %54, ptr %55, align 4
  %56 = load i32, ptr @a28, align 4
  %57 = load ptr, ptr @b28, align 4
  store i32 %56, ptr %57, align 4
  %58 = load i32, ptr @a29, align 4
  %59 = load ptr, ptr @b29, align 4
  store i32 %58, ptr %59, align 4
  %60 = load i32, ptr @a0, align 4
  %61 = load ptr, ptr @c0, align 4
  store i32 %60, ptr %61, align 4
  %62 = load i32, ptr @a1, align 4
  %63 = load ptr, ptr @c1, align 4
  store i32 %62, ptr %63, align 4
  %64 = load i32, ptr @a2, align 4
  %65 = load ptr, ptr @c2, align 4
  store i32 %64, ptr %65, align 4
  %66 = load i32, ptr @a3, align 4
  %67 = load ptr, ptr @c3, align 4
  store i32 %66, ptr %67, align 4
  %68 = load i32, ptr @a4, align 4
  %69 = load ptr, ptr @c4, align 4
  store i32 %68, ptr %69, align 4
  %70 = load i32, ptr @a5, align 4
  %71 = load ptr, ptr @c5, align 4
  store i32 %70, ptr %71, align 4
  %72 = load i32, ptr @a6, align 4
  %73 = load ptr, ptr @c6, align 4
  store i32 %72, ptr %73, align 4
  %74 = load i32, ptr @a7, align 4
  %75 = load ptr, ptr @c7, align 4
  store i32 %74, ptr %75, align 4
  %76 = load i32, ptr @a8, align 4
  %77 = load ptr, ptr @c8, align 4
  store i32 %76, ptr %77, align 4
  %78 = load i32, ptr @a9, align 4
  %79 = load ptr, ptr @c9, align 4
  store i32 %78, ptr %79, align 4
  %80 = load i32, ptr @a10, align 4
  %81 = load ptr, ptr @c10, align 4
  store i32 %80, ptr %81, align 4
  %82 = load i32, ptr @a11, align 4
  %83 = load ptr, ptr @c11, align 4
  store i32 %82, ptr %83, align 4
  %84 = load i32, ptr @a12, align 4
  %85 = load ptr, ptr @c12, align 4
  store i32 %84, ptr %85, align 4
  %86 = load i32, ptr @a13, align 4
  %87 = load ptr, ptr @c13, align 4
  store i32 %86, ptr %87, align 4
  %88 = load i32, ptr @a14, align 4
  %89 = load ptr, ptr @c14, align 4
  store i32 %88, ptr %89, align 4
  %90 = load i32, ptr @a15, align 4
  %91 = load ptr, ptr @c15, align 4
  store i32 %90, ptr %91, align 4
  %92 = load i32, ptr @a16, align 4
  %93 = load ptr, ptr @c16, align 4
  store i32 %92, ptr %93, align 4
  %94 = load i32, ptr @a17, align 4
  %95 = load ptr, ptr @c17, align 4
  store i32 %94, ptr %95, align 4
  %96 = load i32, ptr @a18, align 4
  %97 = load ptr, ptr @c18, align 4
  store i32 %96, ptr %97, align 4
  %98 = load i32, ptr @a19, align 4
  %99 = load ptr, ptr @c19, align 4
  store i32 %98, ptr %99, align 4
  %100 = load i32, ptr @a20, align 4
  %101 = load ptr, ptr @c20, align 4
  store i32 %100, ptr %101, align 4
  %102 = load i32, ptr @a21, align 4
  %103 = load ptr, ptr @c21, align 4
  store i32 %102, ptr %103, align 4
  %104 = load i32, ptr @a22, align 4
  %105 = load ptr, ptr @c22, align 4
  store i32 %104, ptr %105, align 4
  %106 = load i32, ptr @a23, align 4
  %107 = load ptr, ptr @c23, align 4
  store i32 %106, ptr %107, align 4
  %108 = load i32, ptr @a24, align 4
  %109 = load ptr, ptr @c24, align 4
  store i32 %108, ptr %109, align 4
  %110 = load i32, ptr @a25, align 4
  %111 = load ptr, ptr @c25, align 4
  store i32 %110, ptr %111, align 4
  %112 = load i32, ptr @a26, align 4
  %113 = load ptr, ptr @c26, align 4
  store i32 %112, ptr %113, align 4
  %114 = load i32, ptr @a27, align 4
  %115 = load ptr, ptr @c27, align 4
  store i32 %114, ptr %115, align 4
  %116 = load i32, ptr @a28, align 4
  %117 = load ptr, ptr @c28, align 4
  store i32 %116, ptr %117, align 4
  %118 = load i32, ptr @a29, align 4
  %119 = load ptr, ptr @c29, align 4
  store i32 %118, ptr %119, align 4
  %120 = load i32, ptr @a0, align 4
  ret i32 %120
}
