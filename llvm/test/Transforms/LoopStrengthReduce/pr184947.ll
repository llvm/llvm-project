; RUN: opt -S -passes=loop-reduce < %s | FileCheck %s
;
; Make sure loop-reduce doesn't crash with infinite recursion in
; getZeroExtendExpr via getAddExpr calling getZeroExtendExpr without
; propagating the depth argument.
; See https://github.com/llvm/llvm-project/issues/184947
;
; IR produced by clang -O3 -emit-llvm -S from:
;   a() {
;     int b = 0, c = 0;
;     for (;; a) {
;       c++;
;       if (c <= 14) continue;
;       b++;
;       if (b <= 45) continue;
;       return;
;     }
;   }
; The -O3 pipeline unrolls the outer loop, producing 46 chained mini-loops
; whose chain of AddRecs triggered the infinite recursion in SCEV.

; CHECK-LABEL: @a(

define dso_local i32 @a() local_unnamed_addr {
  br label %1
1:                                                ; preds = %0, %1
  %2 = phi i32 [ %3, %1 ], [ 0, %0 ]
  %3 = add nuw nsw i32 %2, 1
  %4 = icmp samesign ult i32 %2, 14
  br i1 %4, label %1, label %5
5:                                                ; preds = %1, %5
  %6 = phi i32 [ %7, %5 ], [ %3, %1 ]
  %7 = add nuw nsw i32 %6, 1
  %8 = icmp samesign ult i32 %6, 14
  br i1 %8, label %5, label %9
9:                                                ; preds = %5, %9
  %10 = phi i32 [ %11, %9 ], [ %7, %5 ]
  %11 = add nuw nsw i32 %10, 1
  %12 = icmp samesign ult i32 %10, 14
  br i1 %12, label %9, label %13
13:                                               ; preds = %9, %13
  %14 = phi i32 [ %15, %13 ], [ %11, %9 ]
  %15 = add nuw nsw i32 %14, 1
  %16 = icmp samesign ult i32 %14, 14
  br i1 %16, label %13, label %17
17:                                               ; preds = %13, %17
  %18 = phi i32 [ %19, %17 ], [ %15, %13 ]
  %19 = add nuw nsw i32 %18, 1
  %20 = icmp samesign ult i32 %18, 14
  br i1 %20, label %17, label %21
21:                                               ; preds = %17, %21
  %22 = phi i32 [ %23, %21 ], [ %19, %17 ]
  %23 = add nuw nsw i32 %22, 1
  %24 = icmp samesign ult i32 %22, 14
  br i1 %24, label %21, label %25
25:                                               ; preds = %21, %25
  %26 = phi i32 [ %27, %25 ], [ %23, %21 ]
  %27 = add nuw nsw i32 %26, 1
  %28 = icmp samesign ult i32 %26, 14
  br i1 %28, label %25, label %29
29:                                               ; preds = %25, %29
  %30 = phi i32 [ %31, %29 ], [ %27, %25 ]
  %31 = add nuw nsw i32 %30, 1
  %32 = icmp samesign ult i32 %30, 14
  br i1 %32, label %29, label %33
33:                                               ; preds = %29, %33
  %34 = phi i32 [ %35, %33 ], [ %31, %29 ]
  %35 = add nuw nsw i32 %34, 1
  %36 = icmp samesign ult i32 %34, 14
  br i1 %36, label %33, label %37
37:                                               ; preds = %33, %37
  %38 = phi i32 [ %39, %37 ], [ %35, %33 ]
  %39 = add nuw nsw i32 %38, 1
  %40 = icmp samesign ult i32 %38, 14
  br i1 %40, label %37, label %41
41:                                               ; preds = %37, %41
  %42 = phi i32 [ %43, %41 ], [ %39, %37 ]
  %43 = add nuw nsw i32 %42, 1
  %44 = icmp samesign ult i32 %42, 14
  br i1 %44, label %41, label %45
45:                                               ; preds = %41, %45
  %46 = phi i32 [ %47, %45 ], [ %43, %41 ]
  %47 = add nuw nsw i32 %46, 1
  %48 = icmp samesign ult i32 %46, 14
  br i1 %48, label %45, label %49
49:                                               ; preds = %45, %49
  %50 = phi i32 [ %51, %49 ], [ %47, %45 ]
  %51 = add nuw nsw i32 %50, 1
  %52 = icmp samesign ult i32 %50, 14
  br i1 %52, label %49, label %53
53:                                               ; preds = %49, %53
  %54 = phi i32 [ %55, %53 ], [ %51, %49 ]
  %55 = add nuw nsw i32 %54, 1
  %56 = icmp samesign ult i32 %54, 14
  br i1 %56, label %53, label %57
57:                                               ; preds = %53, %57
  %58 = phi i32 [ %59, %57 ], [ %55, %53 ]
  %59 = add nuw nsw i32 %58, 1
  %60 = icmp samesign ult i32 %58, 14
  br i1 %60, label %57, label %61
61:                                               ; preds = %57, %61
  %62 = phi i32 [ %63, %61 ], [ %59, %57 ]
  %63 = add nuw nsw i32 %62, 1
  %64 = icmp samesign ult i32 %62, 14
  br i1 %64, label %61, label %65
65:                                               ; preds = %61, %65
  %66 = phi i32 [ %67, %65 ], [ %63, %61 ]
  %67 = add nuw nsw i32 %66, 1
  %68 = icmp samesign ult i32 %66, 14
  br i1 %68, label %65, label %69
69:                                               ; preds = %65, %69
  %70 = phi i32 [ %71, %69 ], [ %67, %65 ]
  %71 = add nuw nsw i32 %70, 1
  %72 = icmp samesign ult i32 %70, 14
  br i1 %72, label %69, label %73
73:                                               ; preds = %69, %73
  %74 = phi i32 [ %75, %73 ], [ %71, %69 ]
  %75 = add nuw nsw i32 %74, 1
  %76 = icmp samesign ult i32 %74, 14
  br i1 %76, label %73, label %77
77:                                               ; preds = %73, %77
  %78 = phi i32 [ %79, %77 ], [ %75, %73 ]
  %79 = add nuw nsw i32 %78, 1
  %80 = icmp samesign ult i32 %78, 14
  br i1 %80, label %77, label %81
81:                                               ; preds = %77, %81
  %82 = phi i32 [ %83, %81 ], [ %79, %77 ]
  %83 = add nuw nsw i32 %82, 1
  %84 = icmp samesign ult i32 %82, 14
  br i1 %84, label %81, label %85
85:                                               ; preds = %81, %85
  %86 = phi i32 [ %87, %85 ], [ %83, %81 ]
  %87 = add nuw nsw i32 %86, 1
  %88 = icmp samesign ult i32 %86, 14
  br i1 %88, label %85, label %89
89:                                               ; preds = %85, %89
  %90 = phi i32 [ %91, %89 ], [ %87, %85 ]
  %91 = add nuw nsw i32 %90, 1
  %92 = icmp samesign ult i32 %90, 14
  br i1 %92, label %89, label %93
93:                                               ; preds = %89, %93
  %94 = phi i32 [ %95, %93 ], [ %91, %89 ]
  %95 = add nuw nsw i32 %94, 1
  %96 = icmp samesign ult i32 %94, 14
  br i1 %96, label %93, label %97
97:                                               ; preds = %93, %97
  %98 = phi i32 [ %99, %97 ], [ %95, %93 ]
  %99 = add nuw nsw i32 %98, 1
  %100 = icmp samesign ult i32 %98, 14
  br i1 %100, label %97, label %101
101:                                              ; preds = %97, %101
  %102 = phi i32 [ %103, %101 ], [ %99, %97 ]
  %103 = add nuw nsw i32 %102, 1
  %104 = icmp samesign ult i32 %102, 14
  br i1 %104, label %101, label %105
105:                                              ; preds = %101, %105
  %106 = phi i32 [ %107, %105 ], [ %103, %101 ]
  %107 = add nuw nsw i32 %106, 1
  %108 = icmp samesign ult i32 %106, 14
  br i1 %108, label %105, label %109
109:                                              ; preds = %105, %109
  %110 = phi i32 [ %111, %109 ], [ %107, %105 ]
  %111 = add nuw nsw i32 %110, 1
  %112 = icmp samesign ult i32 %110, 14
  br i1 %112, label %109, label %113
113:                                              ; preds = %109, %113
  %114 = phi i32 [ %115, %113 ], [ %111, %109 ]
  %115 = add nuw nsw i32 %114, 1
  %116 = icmp samesign ult i32 %114, 14
  br i1 %116, label %113, label %117
117:                                              ; preds = %113, %117
  %118 = phi i32 [ %119, %117 ], [ %115, %113 ]
  %119 = add nuw nsw i32 %118, 1
  %120 = icmp samesign ult i32 %118, 14
  br i1 %120, label %117, label %121
121:                                              ; preds = %117, %121
  %122 = phi i32 [ %123, %121 ], [ %119, %117 ]
  %123 = add nuw nsw i32 %122, 1
  %124 = icmp samesign ult i32 %122, 14
  br i1 %124, label %121, label %125
125:                                              ; preds = %121, %125
  %126 = phi i32 [ %127, %125 ], [ %123, %121 ]
  %127 = add nuw nsw i32 %126, 1
  %128 = icmp samesign ult i32 %126, 14
  br i1 %128, label %125, label %129
129:                                              ; preds = %125, %129
  %130 = phi i32 [ %131, %129 ], [ %127, %125 ]
  %131 = add nuw nsw i32 %130, 1
  %132 = icmp samesign ult i32 %130, 14
  br i1 %132, label %129, label %133
133:                                              ; preds = %129, %133
  %134 = phi i32 [ %135, %133 ], [ %131, %129 ]
  %135 = add nuw nsw i32 %134, 1
  %136 = icmp samesign ult i32 %134, 14
  br i1 %136, label %133, label %137
137:                                              ; preds = %133, %137
  %138 = phi i32 [ %139, %137 ], [ %135, %133 ]
  %139 = add nuw nsw i32 %138, 1
  %140 = icmp samesign ult i32 %138, 14
  br i1 %140, label %137, label %141
141:                                              ; preds = %137, %141
  %142 = phi i32 [ %143, %141 ], [ %139, %137 ]
  %143 = add nuw nsw i32 %142, 1
  %144 = icmp samesign ult i32 %142, 14
  br i1 %144, label %141, label %145
145:                                              ; preds = %141, %145
  %146 = phi i32 [ %147, %145 ], [ %143, %141 ]
  %147 = add nuw nsw i32 %146, 1
  %148 = icmp samesign ult i32 %146, 14
  br i1 %148, label %145, label %149
149:                                              ; preds = %145, %149
  %150 = phi i32 [ %151, %149 ], [ %147, %145 ]
  %151 = add nuw nsw i32 %150, 1
  %152 = icmp samesign ult i32 %150, 14
  br i1 %152, label %149, label %153
153:                                              ; preds = %149, %153
  %154 = phi i32 [ %155, %153 ], [ %151, %149 ]
  %155 = add nuw nsw i32 %154, 1
  %156 = icmp samesign ult i32 %154, 14
  br i1 %156, label %153, label %157
157:                                              ; preds = %153, %157
  %158 = phi i32 [ %159, %157 ], [ %155, %153 ]
  %159 = add nuw nsw i32 %158, 1
  %160 = icmp samesign ult i32 %158, 14
  br i1 %160, label %157, label %161
161:                                              ; preds = %157, %161
  %162 = phi i32 [ %163, %161 ], [ %159, %157 ]
  %163 = add nuw nsw i32 %162, 1
  %164 = icmp samesign ult i32 %162, 14
  br i1 %164, label %161, label %165
165:                                              ; preds = %161, %165
  %166 = phi i32 [ %167, %165 ], [ %163, %161 ]
  %167 = add nuw nsw i32 %166, 1
  %168 = icmp samesign ult i32 %166, 14
  br i1 %168, label %165, label %169
169:                                              ; preds = %165, %169
  %170 = phi i32 [ %171, %169 ], [ %167, %165 ]
  %171 = add nuw nsw i32 %170, 1
  %172 = icmp samesign ult i32 %170, 14
  br i1 %172, label %169, label %173
173:                                              ; preds = %169, %173
  %174 = phi i32 [ %175, %173 ], [ %171, %169 ]
  %175 = add nuw nsw i32 %174, 1
  %176 = icmp samesign ult i32 %174, 14
  br i1 %176, label %173, label %177
177:                                              ; preds = %173, %177
  %178 = phi i32 [ %179, %177 ], [ %175, %173 ]
  %179 = add nuw nsw i32 %178, 1
  %180 = icmp samesign ult i32 %178, 14
  br i1 %180, label %177, label %181
181:                                              ; preds = %177, %181
  %182 = phi i32 [ %183, %181 ], [ %179, %177 ]
  %183 = add nuw nsw i32 %182, 1
  %184 = icmp samesign ult i32 %182, 14
  br i1 %184, label %181, label %185
185:                                              ; preds = %181
  ret i32 undef
}
