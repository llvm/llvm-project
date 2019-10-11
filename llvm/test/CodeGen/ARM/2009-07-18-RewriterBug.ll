; RUN: llc -mtriple armv6-apple-darwin10 -mattr=+vfp2 -filetype asm -o - %s | FileCheck %s

%struct.EDGE_PAIR = type { %struct.edge_rec*, %struct.edge_rec* }
%struct.VEC2 = type { double, double, double }
%struct.VERTEX = type { %struct.VEC2, %struct.VERTEX*, %struct.VERTEX* }
%struct.edge_rec = type { %struct.VERTEX*, %struct.edge_rec*, i32, i8* }
@avail_edge = internal global %struct.edge_rec* null
@_2E_str7 = internal constant [21 x i8] c"ERROR: Only 1 point!\00", section "__TEXT,__cstring,cstring_literals", align 1
@llvm.used = appending global [1 x i8*] [i8* bitcast (void (%struct.EDGE_PAIR*, %struct.VERTEX*, %struct.VERTEX*)* @build_delaunay to i8*)], section "llvm.metadata"

define void @build_delaunay(%struct.EDGE_PAIR* noalias nocapture sret %agg.result, %struct.VERTEX* %tree, %struct.VERTEX* %extra) nounwind {
entry:
  %delright = alloca %struct.EDGE_PAIR, align 8
  %delleft = alloca %struct.EDGE_PAIR, align 8
  %0 = icmp eq %struct.VERTEX* %tree, null
  br i1 %0, label %bb8, label %bb

bb:
  %1 = getelementptr %struct.VERTEX, %struct.VERTEX* %tree, i32 0, i32 2
  %2 = load %struct.VERTEX*, %struct.VERTEX** %1, align 4
  %3 = icmp eq %struct.VERTEX* %2, null
  br i1 %3, label %bb7, label %bb1.i

bb1.i:
  %tree_addr.0.i = phi %struct.VERTEX* [ %5, %bb1.i ], [ %tree, %bb ]
  %4 = getelementptr %struct.VERTEX, %struct.VERTEX* %tree_addr.0.i, i32 0, i32 1
  %5 = load %struct.VERTEX*, %struct.VERTEX** %4, align 4
  %6 = icmp eq %struct.VERTEX* %5, null
  br i1 %6, label %get_low.exit, label %bb1.i

get_low.exit:
  call  void @build_delaunay(%struct.EDGE_PAIR* noalias sret %delright, %struct.VERTEX* %2, %struct.VERTEX* %extra) nounwind
  %7 = getelementptr %struct.VERTEX, %struct.VERTEX* %tree, i32 0, i32 1
  %8 = load %struct.VERTEX*, %struct.VERTEX** %7, align 4
  call  void @build_delaunay(%struct.EDGE_PAIR* noalias sret %delleft, %struct.VERTEX* %8, %struct.VERTEX* %tree) nounwind
  %9 = getelementptr %struct.EDGE_PAIR, %struct.EDGE_PAIR* %delleft, i32 0, i32 0
  %10 = load %struct.edge_rec*, %struct.edge_rec** %9, align 8
  %11 = getelementptr %struct.EDGE_PAIR, %struct.EDGE_PAIR* %delleft, i32 0, i32 1
  %12 = load %struct.edge_rec*, %struct.edge_rec** %11, align 4
  %13 = getelementptr %struct.EDGE_PAIR, %struct.EDGE_PAIR* %delright, i32 0, i32 0
  %14 = load %struct.edge_rec*, %struct.edge_rec** %13, align 8
  %15 = getelementptr %struct.EDGE_PAIR, %struct.EDGE_PAIR* %delright, i32 0, i32 1
  %16 = load %struct.edge_rec*, %struct.edge_rec** %15, align 4
  br label %bb.i

bb.i:
  %rdi_addr.0.i = phi %struct.edge_rec* [ %14, %get_low.exit ], [ %72, %bb4.i ]
  %ldi_addr.1.i = phi %struct.edge_rec* [ %12, %get_low.exit ], [ %ldi_addr.0.i, %bb4.i ]
  %17 = getelementptr %struct.edge_rec, %struct.edge_rec* %rdi_addr.0.i, i32 0, i32 0
  %18 = load %struct.VERTEX*, %struct.VERTEX** %17, align 4
  %19 = ptrtoint %struct.edge_rec* %ldi_addr.1.i to i32
  %20 = getelementptr %struct.VERTEX, %struct.VERTEX* %18, i32 0, i32 0, i32 0
  %21 = load double, double* %20, align 4
  %22 = getelementptr %struct.VERTEX, %struct.VERTEX* %18, i32 0, i32 0, i32 1
  %23 = load double, double* %22, align 4
  br label %bb2.i

bb1.i1:
  %24 = ptrtoint %struct.edge_rec* %ldi_addr.0.i to i32
  %25 = add i32 %24, 48
  %26 = and i32 %25, 63
  %27 = and i32 %24, -64
  %28 = or i32 %26, %27
  %29 = inttoptr i32 %28 to %struct.edge_rec*
  %30 = getelementptr %struct.edge_rec, %struct.edge_rec* %29, i32 0, i32 1
  %31 = load %struct.edge_rec*, %struct.edge_rec** %30, align 4
  %32 = ptrtoint %struct.edge_rec* %31 to i32
  %33 = add i32 %32, 16
  %34 = and i32 %33, 63
  %35 = and i32 %32, -64
  %36 = or i32 %34, %35
  %37 = inttoptr i32 %36 to %struct.edge_rec*
  br label %bb2.i

bb2.i:
  %ldi_addr.1.pn.i = phi %struct.edge_rec* [ %ldi_addr.1.i, %bb.i ], [ %37, %bb1.i1 ]
  %.pn6.in.in.i = phi i32 [ %19, %bb.i ], [ %36, %bb1.i1 ]
  %ldi_addr.0.i = phi %struct.edge_rec* [ %ldi_addr.1.i, %bb.i ], [ %37, %bb1.i1 ]
  %.pn6.in.i = xor i32 %.pn6.in.in.i, 32
  %.pn6.i = inttoptr i32 %.pn6.in.i to %struct.edge_rec*
  %t1.0.in.i = getelementptr %struct.edge_rec, %struct.edge_rec* %ldi_addr.1.pn.i, i32 0, i32 0
  %t2.0.in.i = getelementptr %struct.edge_rec, %struct.edge_rec* %.pn6.i, i32 0, i32 0
  %t1.0.i = load %struct.VERTEX*, %struct.VERTEX** %t1.0.in.i
  %t2.0.i = load %struct.VERTEX*, %struct.VERTEX** %t2.0.in.i
  %38 = getelementptr %struct.VERTEX, %struct.VERTEX* %t1.0.i, i32 0, i32 0, i32 0
  %39 = load double, double* %38, align 4
  %40 = getelementptr %struct.VERTEX, %struct.VERTEX* %t1.0.i, i32 0, i32 0, i32 1
  %41 = load double, double* %40, align 4
  %42 = getelementptr %struct.VERTEX, %struct.VERTEX* %t2.0.i, i32 0, i32 0, i32 0
  %43 = load double, double* %42, align 4
  %44 = getelementptr %struct.VERTEX, %struct.VERTEX* %t2.0.i, i32 0, i32 0, i32 1
  %45 = load double, double* %44, align 4
  %46 = fsub double %39, %21
  %47 = fsub double %45, %23
  %48 = fmul double %46, %47
  %49 = fsub double %43, %21
  %50 = fsub double %41, %23
  %51 = fmul double %49, %50
  %52 = fsub double %48, %51
  %53 = fcmp ogt double %52, 0.000000e+00
  br i1 %53, label %bb1.i1, label %bb3.i

bb3.i:
  %54 = ptrtoint %struct.edge_rec* %rdi_addr.0.i to i32
  %55 = xor i32 %54, 32
  %56 = inttoptr i32 %55 to %struct.edge_rec*
  %57 = getelementptr %struct.edge_rec, %struct.edge_rec* %56, i32 0, i32 0
  %58 = load %struct.VERTEX*, %struct.VERTEX** %57, align 4
  %59 = getelementptr %struct.VERTEX, %struct.VERTEX* %58, i32 0, i32 0, i32 0
  %60 = load double, double* %59, align 4
  %61 = getelementptr %struct.VERTEX, %struct.VERTEX* %58, i32 0, i32 0, i32 1
  %62 = load double, double* %61, align 4
  %63 = fsub double %60, %39
  %64 = fsub double %23, %41
  %65 = fmul double %63, %64
  %66 = fsub double %21, %39
  %67 = fsub double %62, %41
  %68 = fmul double %66, %67
  %69 = fsub double %65, %68
  %70 = fcmp ogt double %69, 0.000000e+00
  br i1 %70, label %bb4.i, label %bb5.i

bb4.i:
  %71 = getelementptr %struct.edge_rec, %struct.edge_rec* %56, i32 0, i32 1
  %72 = load %struct.edge_rec*, %struct.edge_rec** %71, align 4
  br label %bb.i

bb5.i:
  %73 = add i32 %55, 48
  %74 = and i32 %73, 63
  %75 = and i32 %55, -64
  %76 = or i32 %74, %75
  %77 = inttoptr i32 %76 to %struct.edge_rec*
  %78 = getelementptr %struct.edge_rec, %struct.edge_rec* %77, i32 0, i32 1
  %79 = load %struct.edge_rec*, %struct.edge_rec** %78, align 4
  %80 = ptrtoint %struct.edge_rec* %79 to i32
  %81 = add i32 %80, 16
  %82 = and i32 %81, 63
  %83 = and i32 %80, -64
  %84 = or i32 %82, %83
  %85 = inttoptr i32 %84 to %struct.edge_rec*
  %86 = getelementptr %struct.edge_rec, %struct.edge_rec* %ldi_addr.0.i, i32 0, i32 0
  %87 = load %struct.VERTEX*, %struct.VERTEX** %86, align 4
  %88 = call  %struct.edge_rec* @alloc_edge() nounwind
  %89 = getelementptr %struct.edge_rec, %struct.edge_rec* %88, i32 0, i32 1
  store %struct.edge_rec* %88, %struct.edge_rec** %89, align 4
  %90 = getelementptr %struct.edge_rec, %struct.edge_rec* %88, i32 0, i32 0
  store %struct.VERTEX* %18, %struct.VERTEX** %90, align 4
  %91 = ptrtoint %struct.edge_rec* %88 to i32
  %92 = add i32 %91, 16
  %93 = inttoptr i32 %92 to %struct.edge_rec*
  %94 = add i32 %91, 48
  %95 = inttoptr i32 %94 to %struct.edge_rec*
  %96 = getelementptr %struct.edge_rec, %struct.edge_rec* %93, i32 0, i32 1
  store %struct.edge_rec* %95, %struct.edge_rec** %96, align 4
  %97 = add i32 %91, 32
  %98 = inttoptr i32 %97 to %struct.edge_rec*
  %99 = getelementptr %struct.edge_rec, %struct.edge_rec* %98, i32 0, i32 1
  store %struct.edge_rec* %98, %struct.edge_rec** %99, align 4
  %100 = getelementptr %struct.edge_rec, %struct.edge_rec* %98, i32 0, i32 0
  store %struct.VERTEX* %87, %struct.VERTEX** %100, align 4
  %101 = getelementptr %struct.edge_rec, %struct.edge_rec* %95, i32 0, i32 1
  store %struct.edge_rec* %93, %struct.edge_rec** %101, align 4
  %102 = load %struct.edge_rec*, %struct.edge_rec** %89, align 4
  %103 = ptrtoint %struct.edge_rec* %102 to i32
  %104 = add i32 %103, 16
  %105 = and i32 %104, 63
  %106 = and i32 %103, -64
  %107 = or i32 %105, %106
  %108 = inttoptr i32 %107 to %struct.edge_rec*
  %109 = getelementptr %struct.edge_rec, %struct.edge_rec* %85, i32 0, i32 1
  %110 = load %struct.edge_rec*, %struct.edge_rec** %109, align 4
  %111 = ptrtoint %struct.edge_rec* %110 to i32
  %112 = add i32 %111, 16
  %113 = and i32 %112, 63
  %114 = and i32 %111, -64
  %115 = or i32 %113, %114
  %116 = inttoptr i32 %115 to %struct.edge_rec*
  %117 = getelementptr %struct.edge_rec, %struct.edge_rec* %116, i32 0, i32 1
  %118 = load %struct.edge_rec*, %struct.edge_rec** %117, align 4
  %119 = getelementptr %struct.edge_rec, %struct.edge_rec* %108, i32 0, i32 1
  %120 = load %struct.edge_rec*, %struct.edge_rec** %119, align 4
  store %struct.edge_rec* %118, %struct.edge_rec** %119, align 4
  store %struct.edge_rec* %120, %struct.edge_rec** %117, align 4
  %121 = load %struct.edge_rec*, %struct.edge_rec** %89, align 4
  %122 = load %struct.edge_rec*, %struct.edge_rec** %109, align 4
  store %struct.edge_rec* %121, %struct.edge_rec** %109, align 4
  store %struct.edge_rec* %122, %struct.edge_rec** %89, align 4
  %123 = xor i32 %91, 32
  %124 = inttoptr i32 %123 to %struct.edge_rec*
  %125 = getelementptr %struct.edge_rec, %struct.edge_rec* %124, i32 0, i32 1
  %126 = load %struct.edge_rec*, %struct.edge_rec** %125, align 4
  %127 = ptrtoint %struct.edge_rec* %126 to i32
  %128 = add i32 %127, 16
  %129 = and i32 %128, 63
  %130 = and i32 %127, -64
  %131 = or i32 %129, %130
  %132 = inttoptr i32 %131 to %struct.edge_rec*
  %133 = getelementptr %struct.edge_rec, %struct.edge_rec* %ldi_addr.0.i, i32 0, i32 1
  %134 = load %struct.edge_rec*, %struct.edge_rec** %133, align 4
  %135 = ptrtoint %struct.edge_rec* %134 to i32
  %136 = add i32 %135, 16
  %137 = and i32 %136, 63
  %138 = and i32 %135, -64
  %139 = or i32 %137, %138
  %140 = inttoptr i32 %139 to %struct.edge_rec*
  %141 = getelementptr %struct.edge_rec, %struct.edge_rec* %140, i32 0, i32 1
  %142 = load %struct.edge_rec*, %struct.edge_rec** %141, align 4
  %143 = getelementptr %struct.edge_rec, %struct.edge_rec* %132, i32 0, i32 1
  %144 = load %struct.edge_rec*, %struct.edge_rec** %143, align 4
  store %struct.edge_rec* %142, %struct.edge_rec** %143, align 4
  store %struct.edge_rec* %144, %struct.edge_rec** %141, align 4
  %145 = load %struct.edge_rec*, %struct.edge_rec** %125, align 4
  %146 = load %struct.edge_rec*, %struct.edge_rec** %133, align 4
  store %struct.edge_rec* %145, %struct.edge_rec** %133, align 4
  store %struct.edge_rec* %146, %struct.edge_rec** %125, align 4
  %147 = and i32 %92, 63
  %148 = and i32 %91, -64
  %149 = or i32 %147, %148
  %150 = inttoptr i32 %149 to %struct.edge_rec*
  %151 = getelementptr %struct.edge_rec, %struct.edge_rec* %150, i32 0, i32 1
  %152 = load %struct.edge_rec*, %struct.edge_rec** %151, align 4
  %153 = ptrtoint %struct.edge_rec* %152 to i32
  %154 = add i32 %153, 16
  %155 = and i32 %154, 63
  %156 = and i32 %153, -64
  %157 = or i32 %155, %156
  %158 = inttoptr i32 %157 to %struct.edge_rec*
  %159 = load %struct.VERTEX*, %struct.VERTEX** %90, align 4
  %160 = getelementptr %struct.edge_rec, %struct.edge_rec* %124, i32 0, i32 0
  %161 = load %struct.VERTEX*, %struct.VERTEX** %160, align 4
  %162 = getelementptr %struct.edge_rec, %struct.edge_rec* %16, i32 0, i32 0
  %163 = load %struct.VERTEX*, %struct.VERTEX** %162, align 4
  %164 = icmp eq %struct.VERTEX* %163, %159
  %rdo_addr.0.i = select i1 %164, %struct.edge_rec* %88, %struct.edge_rec* %16
  %165 = getelementptr %struct.edge_rec, %struct.edge_rec* %10, i32 0, i32 0
  %166 = load %struct.VERTEX*, %struct.VERTEX** %165, align 4
  %167 = icmp eq %struct.VERTEX* %166, %161
  %ldo_addr.0.ph.i = select i1 %167, %struct.edge_rec* %124, %struct.edge_rec* %10
  br label %bb9.i

bb9.i:
  %lcand.2.i = phi %struct.edge_rec* [ %146, %bb5.i ], [ %lcand.1.i, %bb24.i ], [ %739, %bb25.i ]
  %rcand.2.i = phi %struct.edge_rec* [ %158, %bb5.i ], [ %666, %bb24.i ], [ %rcand.1.i, %bb25.i ]
  %basel.0.i = phi %struct.edge_rec* [ %88, %bb5.i ], [ %595, %bb24.i ], [ %716, %bb25.i ]
  %168 = getelementptr %struct.edge_rec, %struct.edge_rec* %lcand.2.i, i32 0, i32 1
  %169 = load %struct.edge_rec*, %struct.edge_rec** %168, align 4
  %170 = getelementptr %struct.edge_rec, %struct.edge_rec* %basel.0.i, i32 0, i32 0
  %171 = load %struct.VERTEX*, %struct.VERTEX** %170, align 4
  %172 = ptrtoint %struct.edge_rec* %basel.0.i to i32
  %173 = xor i32 %172, 32
  %174 = inttoptr i32 %173 to %struct.edge_rec*
  %175 = getelementptr %struct.edge_rec, %struct.edge_rec* %174, i32 0, i32 0
  %176 = load %struct.VERTEX*, %struct.VERTEX** %175, align 4
  %177 = ptrtoint %struct.edge_rec* %169 to i32
  %178 = xor i32 %177, 32
  %179 = inttoptr i32 %178 to %struct.edge_rec*
  %180 = getelementptr %struct.edge_rec, %struct.edge_rec* %179, i32 0, i32 0
  %181 = load %struct.VERTEX*, %struct.VERTEX** %180, align 4
  %182 = getelementptr %struct.VERTEX, %struct.VERTEX* %171, i32 0, i32 0, i32 0
  %183 = load double, double* %182, align 4
  %184 = getelementptr %struct.VERTEX, %struct.VERTEX* %171, i32 0, i32 0, i32 1
  %185 = load double, double* %184, align 4
  %186 = getelementptr %struct.VERTEX, %struct.VERTEX* %181, i32 0, i32 0, i32 0
  %187 = load double, double* %186, align 4
  %188 = getelementptr %struct.VERTEX, %struct.VERTEX* %181, i32 0, i32 0, i32 1
  %189 = load double, double* %188, align 4
  %190 = getelementptr %struct.VERTEX, %struct.VERTEX* %176, i32 0, i32 0, i32 0
  %191 = load double, double* %190, align 4
  %192 = getelementptr %struct.VERTEX, %struct.VERTEX* %176, i32 0, i32 0, i32 1
  %193 = load double, double* %192, align 4
  %194 = fsub double %183, %191
  %195 = fsub double %189, %193
  %196 = fmul double %194, %195
  %197 = fsub double %187, %191
  %198 = fsub double %185, %193
  %199 = fmul double %197, %198
  %200 = fsub double %196, %199
  %201 = fcmp ogt double %200, 0.000000e+00
  br i1 %201, label %bb10.i, label %bb13.i

bb10.i:
  %202 = getelementptr %struct.VERTEX, %struct.VERTEX* %171, i32 0, i32 0, i32 2
  %avail_edge.promoted25 = load %struct.edge_rec*, %struct.edge_rec** @avail_edge
  br label %bb12.i

bb11.i:
  %203 = ptrtoint %struct.edge_rec* %lcand.0.i to i32
  %204 = add i32 %203, 16
  %205 = and i32 %204, 63
  %206 = and i32 %203, -64
  %207 = or i32 %205, %206
  %208 = inttoptr i32 %207 to %struct.edge_rec*
  %209 = getelementptr %struct.edge_rec, %struct.edge_rec* %208, i32 0, i32 1
  %210 = load %struct.edge_rec*, %struct.edge_rec** %209, align 4
  %211 = ptrtoint %struct.edge_rec* %210 to i32
  %212 = add i32 %211, 16
  %213 = and i32 %212, 63
  %214 = and i32 %211, -64
  %215 = or i32 %213, %214
  %216 = inttoptr i32 %215 to %struct.edge_rec*
  %217 = getelementptr %struct.edge_rec, %struct.edge_rec* %lcand.0.i, i32 0, i32 1
  %218 = load %struct.edge_rec*, %struct.edge_rec** %217, align 4
  %219 = ptrtoint %struct.edge_rec* %218 to i32
  %220 = add i32 %219, 16
  %221 = and i32 %220, 63
  %222 = and i32 %219, -64
  %223 = or i32 %221, %222
  %224 = inttoptr i32 %223 to %struct.edge_rec*
  %225 = getelementptr %struct.edge_rec, %struct.edge_rec* %216, i32 0, i32 1
  %226 = load %struct.edge_rec*, %struct.edge_rec** %225, align 4
  %227 = ptrtoint %struct.edge_rec* %226 to i32
  %228 = add i32 %227, 16
  %229 = and i32 %228, 63
  %230 = and i32 %227, -64
  %231 = or i32 %229, %230
  %232 = inttoptr i32 %231 to %struct.edge_rec*
  %233 = getelementptr %struct.edge_rec, %struct.edge_rec* %232, i32 0, i32 1
  %234 = load %struct.edge_rec*, %struct.edge_rec** %233, align 4
  %235 = getelementptr %struct.edge_rec, %struct.edge_rec* %224, i32 0, i32 1
  %236 = load %struct.edge_rec*, %struct.edge_rec** %235, align 4
  store %struct.edge_rec* %234, %struct.edge_rec** %235, align 4
  store %struct.edge_rec* %236, %struct.edge_rec** %233, align 4
  %237 = load %struct.edge_rec*, %struct.edge_rec** %217, align 4
  %238 = load %struct.edge_rec*, %struct.edge_rec** %225, align 4
  store %struct.edge_rec* %237, %struct.edge_rec** %225, align 4
  store %struct.edge_rec* %238, %struct.edge_rec** %217, align 4
  %239 = xor i32 %203, 32
  %240 = add i32 %239, 16
  %241 = and i32 %240, 63
  %242 = or i32 %241, %206
  %243 = inttoptr i32 %242 to %struct.edge_rec*
  %244 = getelementptr %struct.edge_rec, %struct.edge_rec* %243, i32 0, i32 1
  %245 = load %struct.edge_rec*, %struct.edge_rec** %244, align 4
  %246 = ptrtoint %struct.edge_rec* %245 to i32
  %247 = add i32 %246, 16
  %248 = and i32 %247, 63
  %249 = and i32 %246, -64
  %250 = or i32 %248, %249
  %251 = inttoptr i32 %250 to %struct.edge_rec*
  %252 = inttoptr i32 %239 to %struct.edge_rec*
  %253 = getelementptr %struct.edge_rec, %struct.edge_rec* %252, i32 0, i32 1
  %254 = load %struct.edge_rec*, %struct.edge_rec** %253, align 4
  %255 = ptrtoint %struct.edge_rec* %254 to i32
  %256 = add i32 %255, 16
  %257 = and i32 %256, 63
  %258 = and i32 %255, -64
  %259 = or i32 %257, %258
  %260 = inttoptr i32 %259 to %struct.edge_rec*
  %261 = getelementptr %struct.edge_rec, %struct.edge_rec* %251, i32 0, i32 1
  %262 = load %struct.edge_rec*, %struct.edge_rec** %261, align 4
  %263 = ptrtoint %struct.edge_rec* %262 to i32
  %264 = add i32 %263, 16
  %265 = and i32 %264, 63
  %266 = and i32 %263, -64
  %267 = or i32 %265, %266
  %268 = inttoptr i32 %267 to %struct.edge_rec*
  %269 = getelementptr %struct.edge_rec, %struct.edge_rec* %268, i32 0, i32 1
  %270 = load %struct.edge_rec*, %struct.edge_rec** %269, align 4
  %271 = getelementptr %struct.edge_rec, %struct.edge_rec* %260, i32 0, i32 1
  %272 = load %struct.edge_rec*, %struct.edge_rec** %271, align 4
  store %struct.edge_rec* %270, %struct.edge_rec** %271, align 4
  store %struct.edge_rec* %272, %struct.edge_rec** %269, align 4
  %273 = load %struct.edge_rec*, %struct.edge_rec** %253, align 4
  %274 = load %struct.edge_rec*, %struct.edge_rec** %261, align 4
  store %struct.edge_rec* %273, %struct.edge_rec** %261, align 4
  store %struct.edge_rec* %274, %struct.edge_rec** %253, align 4
  %275 = inttoptr i32 %206 to %struct.edge_rec*
  %276 = getelementptr %struct.edge_rec, %struct.edge_rec* %275, i32 0, i32 1
  store %struct.edge_rec* %avail_edge.tmp.026, %struct.edge_rec** %276, align 4
  %277 = getelementptr %struct.edge_rec, %struct.edge_rec* %t.0.i, i32 0, i32 1
  %278 = load %struct.edge_rec*, %struct.edge_rec** %277, align 4
  %.pre.i = load double, double* %182, align 4
  %.pre22.i = load double, double* %184, align 4
  br label %bb12.i

bb12.i:
  %avail_edge.tmp.026 = phi %struct.edge_rec* [ %avail_edge.promoted25, %bb10.i ], [ %275, %bb11.i ]
  %279 = phi double [ %.pre22.i, %bb11.i ], [ %185, %bb10.i ]
  %280 = phi double [ %.pre.i, %bb11.i ], [ %183, %bb10.i ]
  %lcand.0.i = phi %struct.edge_rec* [ %lcand.2.i, %bb10.i ], [ %t.0.i, %bb11.i ]
  %t.0.i = phi %struct.edge_rec* [ %169, %bb10.i ], [ %278, %bb11.i ]
  %.pn5.in.in.in.i = phi %struct.edge_rec* [ %lcand.2.i, %bb10.i ], [ %t.0.i, %bb11.i ]
  %.pn4.in.in.in.i = phi %struct.edge_rec* [ %169, %bb10.i ], [ %278, %bb11.i ]
  %lcand.2.pn.i = phi %struct.edge_rec* [ %lcand.2.i, %bb10.i ], [ %t.0.i, %bb11.i ]
  %.pn5.in.in.i = ptrtoint %struct.edge_rec* %.pn5.in.in.in.i to i32
  %.pn4.in.in.i = ptrtoint %struct.edge_rec* %.pn4.in.in.in.i to i32
  %.pn5.in.i = xor i32 %.pn5.in.in.i, 32
  %.pn4.in.i = xor i32 %.pn4.in.in.i, 32
  %.pn5.i = inttoptr i32 %.pn5.in.i to %struct.edge_rec*
  %.pn4.i = inttoptr i32 %.pn4.in.i to %struct.edge_rec*
  %v1.0.in.i = getelementptr %struct.edge_rec, %struct.edge_rec* %.pn5.i, i32 0, i32 0
  %v2.0.in.i = getelementptr %struct.edge_rec, %struct.edge_rec* %.pn4.i, i32 0, i32 0
  %v3.0.in.i = getelementptr %struct.edge_rec, %struct.edge_rec* %lcand.2.pn.i, i32 0, i32 0
  %v1.0.i = load %struct.VERTEX*, %struct.VERTEX** %v1.0.in.i
  %v2.0.i = load %struct.VERTEX*, %struct.VERTEX** %v2.0.in.i
  %v3.0.i = load %struct.VERTEX*, %struct.VERTEX** %v3.0.in.i
  %281 = load double, double* %202, align 4
  %282 = getelementptr %struct.VERTEX, %struct.VERTEX* %v1.0.i, i32 0, i32 0, i32 0
  %283 = load double, double* %282, align 4
  %284 = fsub double %283, %280
  %285 = getelementptr %struct.VERTEX, %struct.VERTEX* %v1.0.i, i32 0, i32 0, i32 1
  %286 = load double, double* %285, align 4
  %287 = fsub double %286, %279
  %288 = getelementptr %struct.VERTEX, %struct.VERTEX* %v1.0.i, i32 0, i32 0, i32 2
  %289 = load double, double* %288, align 4
  %290 = getelementptr %struct.VERTEX, %struct.VERTEX* %v2.0.i, i32 0, i32 0, i32 0
  %291 = load double, double* %290, align 4
  %292 = fsub double %291, %280
  %293 = getelementptr %struct.VERTEX, %struct.VERTEX* %v2.0.i, i32 0, i32 0, i32 1
  %294 = load double, double* %293, align 4
  %295 = fsub double %294, %279
  %296 = getelementptr %struct.VERTEX, %struct.VERTEX* %v2.0.i, i32 0, i32 0, i32 2
  %297 = load double, double* %296, align 4
  %298 = getelementptr %struct.VERTEX, %struct.VERTEX* %v3.0.i, i32 0, i32 0, i32 0
  %299 = load double, double* %298, align 4
  %300 = fsub double %299, %280
  %301 = getelementptr %struct.VERTEX, %struct.VERTEX* %v3.0.i, i32 0, i32 0, i32 1
  %302 = load double, double* %301, align 4
  %303 = fsub double %302, %279
  %304 = getelementptr %struct.VERTEX, %struct.VERTEX* %v3.0.i, i32 0, i32 0, i32 2
  %305 = load double, double* %304, align 4
  %306 = fsub double %289, %281
  %307 = fmul double %292, %303
  %308 = fmul double %295, %300
  %309 = fsub double %307, %308
  %310 = fmul double %306, %309
  %311 = fsub double %297, %281
  %312 = fmul double %300, %287
  %313 = fmul double %303, %284
  %314 = fsub double %312, %313
  %315 = fmul double %311, %314
  %316 = fadd double %315, %310
  %317 = fsub double %305, %281
  %318 = fmul double %284, %295
  %319 = fmul double %287, %292
  %320 = fsub double %318, %319
  %321 = fmul double %317, %320
  %322 = fadd double %321, %316
  %323 = fcmp ogt double %322, 0.000000e+00
  br i1 %323, label %bb11.i, label %bb13.loopexit.i

bb13.loopexit.i:
  store %struct.edge_rec* %avail_edge.tmp.026, %struct.edge_rec** @avail_edge
  %.pre23.i = load %struct.VERTEX*, %struct.VERTEX** %170, align 4
  %.pre24.i = load %struct.VERTEX*, %struct.VERTEX** %175, align 4
  br label %bb13.i

bb13.i:
  %324 = phi %struct.VERTEX* [ %.pre24.i, %bb13.loopexit.i ], [ %176, %bb9.i ]
  %325 = phi %struct.VERTEX* [ %.pre23.i, %bb13.loopexit.i ], [ %171, %bb9.i ]
  %lcand.1.i = phi %struct.edge_rec* [ %lcand.0.i, %bb13.loopexit.i ], [ %lcand.2.i, %bb9.i ]
  %326 = ptrtoint %struct.edge_rec* %rcand.2.i to i32
  %327 = add i32 %326, 16
  %328 = and i32 %327, 63
  %329 = and i32 %326, -64
  %330 = or i32 %328, %329
  %331 = inttoptr i32 %330 to %struct.edge_rec*
  %332 = getelementptr %struct.edge_rec, %struct.edge_rec* %331, i32 0, i32 1
  %333 = load %struct.edge_rec*, %struct.edge_rec** %332, align 4
  %334 = ptrtoint %struct.edge_rec* %333 to i32
  %335 = add i32 %334, 16
  %336 = and i32 %335, 63
  %337 = and i32 %334, -64
  %338 = or i32 %336, %337
  %339 = xor i32 %338, 32
  %340 = inttoptr i32 %339 to %struct.edge_rec*
  %341 = getelementptr %struct.edge_rec, %struct.edge_rec* %340, i32 0, i32 0
  %342 = load %struct.VERTEX*, %struct.VERTEX** %341, align 4
  %343 = getelementptr %struct.VERTEX, %struct.VERTEX* %325, i32 0, i32 0, i32 0
  %344 = load double, double* %343, align 4
  %345 = getelementptr %struct.VERTEX, %struct.VERTEX* %325, i32 0, i32 0, i32 1
  %346 = load double, double* %345, align 4
  %347 = getelementptr %struct.VERTEX, %struct.VERTEX* %342, i32 0, i32 0, i32 0
  %348 = load double, double* %347, align 4
  %349 = getelementptr %struct.VERTEX, %struct.VERTEX* %342, i32 0, i32 0, i32 1
  %350 = load double, double* %349, align 4
  %351 = getelementptr %struct.VERTEX, %struct.VERTEX* %324, i32 0, i32 0, i32 0
  %352 = load double, double* %351, align 4
  %353 = getelementptr %struct.VERTEX, %struct.VERTEX* %324, i32 0, i32 0, i32 1
  %354 = load double, double* %353, align 4
  %355 = fsub double %344, %352
  %356 = fsub double %350, %354
  %357 = fmul double %355, %356
  %358 = fsub double %348, %352
  %359 = fsub double %346, %354
  %360 = fmul double %358, %359
  %361 = fsub double %357, %360
  %362 = fcmp ogt double %361, 0.000000e+00
  br i1 %362, label %bb14.i, label %bb17.i

bb14.i:
  %363 = getelementptr %struct.VERTEX, %struct.VERTEX* %324, i32 0, i32 0, i32 2
  %avail_edge.promoted = load %struct.edge_rec*, %struct.edge_rec** @avail_edge
  br label %bb16.i

bb15.i:
  %364 = ptrtoint %struct.edge_rec* %rcand.0.i to i32
  %365 = add i32 %364, 16
  %366 = and i32 %365, 63
  %367 = and i32 %364, -64
  %368 = or i32 %366, %367
  %369 = inttoptr i32 %368 to %struct.edge_rec*
  %370 = getelementptr %struct.edge_rec, %struct.edge_rec* %369, i32 0, i32 1
  %371 = load %struct.edge_rec*, %struct.edge_rec** %370, align 4
  %372 = ptrtoint %struct.edge_rec* %371 to i32
  %373 = add i32 %372, 16
  %374 = and i32 %373, 63
  %375 = and i32 %372, -64
  %376 = or i32 %374, %375
  %377 = inttoptr i32 %376 to %struct.edge_rec*
  %378 = getelementptr %struct.edge_rec, %struct.edge_rec* %rcand.0.i, i32 0, i32 1
  %379 = load %struct.edge_rec*, %struct.edge_rec** %378, align 4
  %380 = ptrtoint %struct.edge_rec* %379 to i32
  %381 = add i32 %380, 16
  %382 = and i32 %381, 63
  %383 = and i32 %380, -64
  %384 = or i32 %382, %383
  %385 = inttoptr i32 %384 to %struct.edge_rec*
  %386 = getelementptr %struct.edge_rec, %struct.edge_rec* %377, i32 0, i32 1
  %387 = load %struct.edge_rec*, %struct.edge_rec** %386, align 4
  %388 = ptrtoint %struct.edge_rec* %387 to i32
  %389 = add i32 %388, 16
  %390 = and i32 %389, 63
  %391 = and i32 %388, -64
  %392 = or i32 %390, %391
  %393 = inttoptr i32 %392 to %struct.edge_rec*
  %394 = getelementptr %struct.edge_rec, %struct.edge_rec* %393, i32 0, i32 1
  %395 = load %struct.edge_rec*, %struct.edge_rec** %394, align 4
  %396 = getelementptr %struct.edge_rec, %struct.edge_rec* %385, i32 0, i32 1
  %397 = load %struct.edge_rec*, %struct.edge_rec** %396, align 4
  store %struct.edge_rec* %395, %struct.edge_rec** %396, align 4
  store %struct.edge_rec* %397, %struct.edge_rec** %394, align 4
  %398 = load %struct.edge_rec*, %struct.edge_rec** %378, align 4
  %399 = load %struct.edge_rec*, %struct.edge_rec** %386, align 4
  store %struct.edge_rec* %398, %struct.edge_rec** %386, align 4
  store %struct.edge_rec* %399, %struct.edge_rec** %378, align 4
  %400 = xor i32 %364, 32
  %401 = add i32 %400, 16
  %402 = and i32 %401, 63
  %403 = or i32 %402, %367
  %404 = inttoptr i32 %403 to %struct.edge_rec*
  %405 = getelementptr %struct.edge_rec, %struct.edge_rec* %404, i32 0, i32 1
  %406 = load %struct.edge_rec*, %struct.edge_rec** %405, align 4
  %407 = ptrtoint %struct.edge_rec* %406 to i32
  %408 = add i32 %407, 16
  %409 = and i32 %408, 63
  %410 = and i32 %407, -64
  %411 = or i32 %409, %410
  %412 = inttoptr i32 %411 to %struct.edge_rec*
  %413 = inttoptr i32 %400 to %struct.edge_rec*
  %414 = getelementptr %struct.edge_rec, %struct.edge_rec* %413, i32 0, i32 1
  %415 = load %struct.edge_rec*, %struct.edge_rec** %414, align 4
  %416 = ptrtoint %struct.edge_rec* %415 to i32
  %417 = add i32 %416, 16
  %418 = and i32 %417, 63
  %419 = and i32 %416, -64
  %420 = or i32 %418, %419
  %421 = inttoptr i32 %420 to %struct.edge_rec*
  %422 = getelementptr %struct.edge_rec, %struct.edge_rec* %412, i32 0, i32 1
  %423 = load %struct.edge_rec*, %struct.edge_rec** %422, align 4
  %424 = ptrtoint %struct.edge_rec* %423 to i32
  %425 = add i32 %424, 16
  %426 = and i32 %425, 63
  %427 = and i32 %424, -64
  %428 = or i32 %426, %427
  %429 = inttoptr i32 %428 to %struct.edge_rec*
  %430 = getelementptr %struct.edge_rec, %struct.edge_rec* %429, i32 0, i32 1
  %431 = load %struct.edge_rec*, %struct.edge_rec** %430, align 4
  %432 = getelementptr %struct.edge_rec, %struct.edge_rec* %421, i32 0, i32 1
  %433 = load %struct.edge_rec*, %struct.edge_rec** %432, align 4
  store %struct.edge_rec* %431, %struct.edge_rec** %432, align 4
  store %struct.edge_rec* %433, %struct.edge_rec** %430, align 4
  %434 = load %struct.edge_rec*, %struct.edge_rec** %414, align 4
  %435 = load %struct.edge_rec*, %struct.edge_rec** %422, align 4
  store %struct.edge_rec* %434, %struct.edge_rec** %422, align 4
  store %struct.edge_rec* %435, %struct.edge_rec** %414, align 4
  %436 = inttoptr i32 %367 to %struct.edge_rec*
  %437 = getelementptr %struct.edge_rec, %struct.edge_rec* %436, i32 0, i32 1
  store %struct.edge_rec* %avail_edge.tmp.0, %struct.edge_rec** %437, align 4
  %438 = add i32 %t.1.in.i, 16
  %439 = and i32 %438, 63
  %440 = and i32 %t.1.in.i, -64
  %441 = or i32 %439, %440
  %442 = inttoptr i32 %441 to %struct.edge_rec*
  %443 = getelementptr %struct.edge_rec, %struct.edge_rec* %442, i32 0, i32 1
  %444 = load %struct.edge_rec*, %struct.edge_rec** %443, align 4
  %445 = ptrtoint %struct.edge_rec* %444 to i32
  %446 = add i32 %445, 16
  %447 = and i32 %446, 63
  %448 = and i32 %445, -64
  %449 = or i32 %447, %448
  %.pre25.i = load double, double* %351, align 4
  %.pre26.i = load double, double* %353, align 4
  br label %bb16.i

bb16.i:
  %avail_edge.tmp.0 = phi %struct.edge_rec* [ %avail_edge.promoted, %bb14.i ], [ %436, %bb15.i ]
  %450 = phi double [ %.pre26.i, %bb15.i ], [ %354, %bb14.i ]
  %451 = phi double [ %.pre25.i, %bb15.i ], [ %352, %bb14.i ]
  %rcand.0.i = phi %struct.edge_rec* [ %rcand.2.i, %bb14.i ], [ %t.1.i, %bb15.i ]
  %t.1.in.i = phi i32 [ %338, %bb14.i ], [ %449, %bb15.i ]
  %.pn3.in.in.i = phi i32 [ %338, %bb14.i ], [ %449, %bb15.i ]
  %.pn.in.in.in.i = phi %struct.edge_rec* [ %rcand.2.i, %bb14.i ], [ %t.1.i, %bb15.i ]
  %rcand.2.pn.i = phi %struct.edge_rec* [ %rcand.2.i, %bb14.i ], [ %t.1.i, %bb15.i ]
  %t.1.i = inttoptr i32 %t.1.in.i to %struct.edge_rec*
  %.pn.in.in.i = ptrtoint %struct.edge_rec* %.pn.in.in.in.i to i32
  %.pn3.in.i = xor i32 %.pn3.in.in.i, 32
  %.pn.in.i = xor i32 %.pn.in.in.i, 32
  %.pn3.i = inttoptr i32 %.pn3.in.i to %struct.edge_rec*
  %.pn.i = inttoptr i32 %.pn.in.i to %struct.edge_rec*
  %v1.1.in.i = getelementptr %struct.edge_rec, %struct.edge_rec* %.pn3.i, i32 0, i32 0
  %v2.1.in.i = getelementptr %struct.edge_rec, %struct.edge_rec* %.pn.i, i32 0, i32 0
  %v3.1.in.i = getelementptr %struct.edge_rec, %struct.edge_rec* %rcand.2.pn.i, i32 0, i32 0
  %v1.1.i = load %struct.VERTEX*, %struct.VERTEX** %v1.1.in.i
  %v2.1.i = load %struct.VERTEX*, %struct.VERTEX** %v2.1.in.i
  %v3.1.i = load %struct.VERTEX*, %struct.VERTEX** %v3.1.in.i
  %452 = load double, double* %363, align 4
  %453 = getelementptr %struct.VERTEX, %struct.VERTEX* %v1.1.i, i32 0, i32 0, i32 0
  %454 = load double, double* %453, align 4
  %455 = fsub double %454, %451
  %456 = getelementptr %struct.VERTEX, %struct.VERTEX* %v1.1.i, i32 0, i32 0, i32 1
  %457 = load double, double* %456, align 4
  %458 = fsub double %457, %450
  %459 = getelementptr %struct.VERTEX, %struct.VERTEX* %v1.1.i, i32 0, i32 0, i32 2
  %460 = load double, double* %459, align 4
  %461 = getelementptr %struct.VERTEX, %struct.VERTEX* %v2.1.i, i32 0, i32 0, i32 0
  %462 = load double, double* %461, align 4
  %463 = fsub double %462, %451
  %464 = getelementptr %struct.VERTEX, %struct.VERTEX* %v2.1.i, i32 0, i32 0, i32 1
  %465 = load double, double* %464, align 4
  %466 = fsub double %465, %450
  %467 = getelementptr %struct.VERTEX, %struct.VERTEX* %v2.1.i, i32 0, i32 0, i32 2
  %468 = load double, double* %467, align 4
  %469 = getelementptr %struct.VERTEX, %struct.VERTEX* %v3.1.i, i32 0, i32 0, i32 0
  %470 = load double, double* %469, align 4
  %471 = fsub double %470, %451
  %472 = getelementptr %struct.VERTEX, %struct.VERTEX* %v3.1.i, i32 0, i32 0, i32 1
  %473 = load double, double* %472, align 4
  %474 = fsub double %473, %450
  %475 = getelementptr %struct.VERTEX, %struct.VERTEX* %v3.1.i, i32 0, i32 0, i32 2
  %476 = load double, double* %475, align 4
  %477 = fsub double %460, %452
  %478 = fmul double %463, %474
  %479 = fmul double %466, %471
  %480 = fsub double %478, %479
  %481 = fmul double %477, %480
  %482 = fsub double %468, %452
  %483 = fmul double %471, %458
  %484 = fmul double %474, %455
  %485 = fsub double %483, %484
  %486 = fmul double %482, %485
  %487 = fadd double %486, %481
  %488 = fsub double %476, %452
  %489 = fmul double %455, %466
  %490 = fmul double %458, %463
  %491 = fsub double %489, %490
  %492 = fmul double %488, %491
  %493 = fadd double %492, %487
  %494 = fcmp ogt double %493, 0.000000e+00
  br i1 %494, label %bb15.i, label %bb17.loopexit.i

bb17.loopexit.i:
  store %struct.edge_rec* %avail_edge.tmp.0, %struct.edge_rec** @avail_edge
  %.pre27.i = load %struct.VERTEX*, %struct.VERTEX** %170, align 4
  %.pre28.i = load %struct.VERTEX*, %struct.VERTEX** %175, align 4
  br label %bb17.i

bb17.i:
  %495 = phi %struct.VERTEX* [ %.pre28.i, %bb17.loopexit.i ], [ %324, %bb13.i ]
  %496 = phi %struct.VERTEX* [ %.pre27.i, %bb17.loopexit.i ], [ %325, %bb13.i ]
  %rcand.1.i = phi %struct.edge_rec* [ %rcand.0.i, %bb17.loopexit.i ], [ %rcand.2.i, %bb13.i ]
  %497 = ptrtoint %struct.edge_rec* %lcand.1.i to i32
  %498 = xor i32 %497, 32
  %499 = inttoptr i32 %498 to %struct.edge_rec*
  %500 = getelementptr %struct.edge_rec, %struct.edge_rec* %499, i32 0, i32 0
  %501 = load %struct.VERTEX*, %struct.VERTEX** %500, align 4
  %502 = getelementptr %struct.VERTEX, %struct.VERTEX* %496, i32 0, i32 0, i32 0
  %503 = load double, double* %502, align 4
  %504 = getelementptr %struct.VERTEX, %struct.VERTEX* %496, i32 0, i32 0, i32 1
  %505 = load double, double* %504, align 4
  %506 = getelementptr %struct.VERTEX, %struct.VERTEX* %501, i32 0, i32 0, i32 0
  %507 = load double, double* %506, align 4
  %508 = getelementptr %struct.VERTEX, %struct.VERTEX* %501, i32 0, i32 0, i32 1
  %509 = load double, double* %508, align 4
  %510 = getelementptr %struct.VERTEX, %struct.VERTEX* %495, i32 0, i32 0, i32 0
  %511 = load double, double* %510, align 4
  %512 = getelementptr %struct.VERTEX, %struct.VERTEX* %495, i32 0, i32 0, i32 1
  %513 = load double, double* %512, align 4
  %514 = fsub double %503, %511
  %515 = fsub double %509, %513
  %516 = fmul double %514, %515
  %517 = fsub double %507, %511
  %518 = fsub double %505, %513
  %519 = fmul double %517, %518
  %520 = fsub double %516, %519
  %521 = fcmp ogt double %520, 0.000000e+00
  %522 = ptrtoint %struct.edge_rec* %rcand.1.i to i32
  %523 = xor i32 %522, 32
  %524 = inttoptr i32 %523 to %struct.edge_rec*
  %525 = getelementptr %struct.edge_rec, %struct.edge_rec* %524, i32 0, i32 0
  %526 = load %struct.VERTEX*, %struct.VERTEX** %525, align 4
  %527 = getelementptr %struct.VERTEX, %struct.VERTEX* %526, i32 0, i32 0, i32 0
  %528 = load double, double* %527, align 4
  %529 = getelementptr %struct.VERTEX, %struct.VERTEX* %526, i32 0, i32 0, i32 1
  %530 = load double, double* %529, align 4
  %531 = fsub double %530, %513
  %532 = fmul double %514, %531
  %533 = fsub double %528, %511
  %534 = fmul double %533, %518
  %535 = fsub double %532, %534
  %536 = fcmp ogt double %535, 0.000000e+00
  %537 = or i1 %536, %521
  br i1 %537, label %bb21.i, label %do_merge.exit

bb21.i:
  %538 = getelementptr %struct.edge_rec, %struct.edge_rec* %lcand.1.i, i32 0, i32 0
  %539 = load %struct.VERTEX*, %struct.VERTEX** %538, align 4
  %540 = getelementptr %struct.edge_rec, %struct.edge_rec* %rcand.1.i, i32 0, i32 0
  %541 = load %struct.VERTEX*, %struct.VERTEX** %540, align 4
  br i1 %521, label %bb22.i, label %bb24.i

bb22.i:
  br i1 %536, label %bb23.i, label %bb25.i

bb23.i:
  %542 = getelementptr %struct.VERTEX, %struct.VERTEX* %526, i32 0, i32 0, i32 2
  %543 = load double, double* %542, align 4
  %544 = fsub double %507, %528
  %545 = fsub double %509, %530
  %546 = getelementptr %struct.VERTEX, %struct.VERTEX* %501, i32 0, i32 0, i32 2
  %547 = load double, double* %546, align 4
  %548 = getelementptr %struct.VERTEX, %struct.VERTEX* %539, i32 0, i32 0, i32 0
  %549 = load double, double* %548, align 4
  %550 = fsub double %549, %528
  %551 = getelementptr %struct.VERTEX, %struct.VERTEX* %539, i32 0, i32 0, i32 1
  %552 = load double, double* %551, align 4
  %553 = fsub double %552, %530
  %554 = getelementptr %struct.VERTEX, %struct.VERTEX* %539, i32 0, i32 0, i32 2
  %555 = load double, double* %554, align 4
  %556 = getelementptr %struct.VERTEX, %struct.VERTEX* %541, i32 0, i32 0, i32 0
  %557 = load double, double* %556, align 4
  %558 = fsub double %557, %528
  %559 = getelementptr %struct.VERTEX, %struct.VERTEX* %541, i32 0, i32 0, i32 1
  %560 = load double, double* %559, align 4
  %561 = fsub double %560, %530
  %562 = getelementptr %struct.VERTEX, %struct.VERTEX* %541, i32 0, i32 0, i32 2
  %563 = load double, double* %562, align 4
  %564 = fsub double %547, %543
  %565 = fmul double %550, %561
  %566 = fmul double %553, %558
  %567 = fsub double %565, %566
  %568 = fmul double %564, %567
  %569 = fsub double %555, %543
  %570 = fmul double %558, %545
  %571 = fmul double %561, %544
  %572 = fsub double %570, %571
  %573 = fmul double %569, %572
  %574 = fadd double %573, %568
  %575 = fsub double %563, %543
  %576 = fmul double %544, %553
  %577 = fmul double %545, %550
  %578 = fsub double %576, %577
  %579 = fmul double %575, %578
  %580 = fadd double %579, %574
  %581 = fcmp ogt double %580, 0.000000e+00
  br i1 %581, label %bb24.i, label %bb25.i

bb24.i:
  %582 = add i32 %522, 48
  %583 = and i32 %582, 63
  %584 = and i32 %522, -64
  %585 = or i32 %583, %584
  %586 = inttoptr i32 %585 to %struct.edge_rec*
  %587 = getelementptr %struct.edge_rec, %struct.edge_rec* %586, i32 0, i32 1
  %588 = load %struct.edge_rec*, %struct.edge_rec** %587, align 4
  %589 = ptrtoint %struct.edge_rec* %588 to i32
  %590 = add i32 %589, 16
  %591 = and i32 %590, 63
  %592 = and i32 %589, -64
  %593 = or i32 %591, %592
  %594 = inttoptr i32 %593 to %struct.edge_rec*
  %595 = call  %struct.edge_rec* @alloc_edge() nounwind
  %596 = getelementptr %struct.edge_rec, %struct.edge_rec* %595, i32 0, i32 1
  store %struct.edge_rec* %595, %struct.edge_rec** %596, align 4
  %597 = getelementptr %struct.edge_rec, %struct.edge_rec* %595, i32 0, i32 0
  store %struct.VERTEX* %526, %struct.VERTEX** %597, align 4
  %598 = ptrtoint %struct.edge_rec* %595 to i32
  %599 = add i32 %598, 16
  %600 = inttoptr i32 %599 to %struct.edge_rec*
  %601 = add i32 %598, 48
  %602 = inttoptr i32 %601 to %struct.edge_rec*
  %603 = getelementptr %struct.edge_rec, %struct.edge_rec* %600, i32 0, i32 1
  store %struct.edge_rec* %602, %struct.edge_rec** %603, align 4
  %604 = add i32 %598, 32
  %605 = inttoptr i32 %604 to %struct.edge_rec*
  %606 = getelementptr %struct.edge_rec, %struct.edge_rec* %605, i32 0, i32 1
  store %struct.edge_rec* %605, %struct.edge_rec** %606, align 4
  %607 = getelementptr %struct.edge_rec, %struct.edge_rec* %605, i32 0, i32 0
  store %struct.VERTEX* %495, %struct.VERTEX** %607, align 4
  %608 = getelementptr %struct.edge_rec, %struct.edge_rec* %602, i32 0, i32 1
  store %struct.edge_rec* %600, %struct.edge_rec** %608, align 4
  %609 = load %struct.edge_rec*, %struct.edge_rec** %596, align 4
  %610 = ptrtoint %struct.edge_rec* %609 to i32
  %611 = add i32 %610, 16
  %612 = and i32 %611, 63
  %613 = and i32 %610, -64
  %614 = or i32 %612, %613
  %615 = inttoptr i32 %614 to %struct.edge_rec*
  %616 = getelementptr %struct.edge_rec, %struct.edge_rec* %594, i32 0, i32 1
  %617 = load %struct.edge_rec*, %struct.edge_rec** %616, align 4
  %618 = ptrtoint %struct.edge_rec* %617 to i32
  %619 = add i32 %618, 16
  %620 = and i32 %619, 63
  %621 = and i32 %618, -64
  %622 = or i32 %620, %621
  %623 = inttoptr i32 %622 to %struct.edge_rec*
  %624 = getelementptr %struct.edge_rec, %struct.edge_rec* %623, i32 0, i32 1
  %625 = load %struct.edge_rec*, %struct.edge_rec** %624, align 4
  %626 = getelementptr %struct.edge_rec, %struct.edge_rec* %615, i32 0, i32 1
  %627 = load %struct.edge_rec*, %struct.edge_rec** %626, align 4
  store %struct.edge_rec* %625, %struct.edge_rec** %626, align 4
  store %struct.edge_rec* %627, %struct.edge_rec** %624, align 4
  %628 = load %struct.edge_rec*, %struct.edge_rec** %596, align 4
  %629 = load %struct.edge_rec*, %struct.edge_rec** %616, align 4
  store %struct.edge_rec* %628, %struct.edge_rec** %616, align 4
  store %struct.edge_rec* %629, %struct.edge_rec** %596, align 4
  %630 = xor i32 %598, 32
  %631 = inttoptr i32 %630 to %struct.edge_rec*
  %632 = getelementptr %struct.edge_rec, %struct.edge_rec* %631, i32 0, i32 1
  %633 = load %struct.edge_rec*, %struct.edge_rec** %632, align 4
  %634 = ptrtoint %struct.edge_rec* %633 to i32
  %635 = add i32 %634, 16
  %636 = and i32 %635, 63
  %637 = and i32 %634, -64
  %638 = or i32 %636, %637
  %639 = inttoptr i32 %638 to %struct.edge_rec*
  %640 = getelementptr %struct.edge_rec, %struct.edge_rec* %174, i32 0, i32 1
  %641 = load %struct.edge_rec*, %struct.edge_rec** %640, align 4
  %642 = ptrtoint %struct.edge_rec* %641 to i32
  %643 = add i32 %642, 16
  %644 = and i32 %643, 63
  %645 = and i32 %642, -64
  %646 = or i32 %644, %645
  %647 = inttoptr i32 %646 to %struct.edge_rec*
  %648 = getelementptr %struct.edge_rec, %struct.edge_rec* %647, i32 0, i32 1
  %649 = load %struct.edge_rec*, %struct.edge_rec** %648, align 4
  %650 = getelementptr %struct.edge_rec, %struct.edge_rec* %639, i32 0, i32 1
  %651 = load %struct.edge_rec*, %struct.edge_rec** %650, align 4
  store %struct.edge_rec* %649, %struct.edge_rec** %650, align 4
  store %struct.edge_rec* %651, %struct.edge_rec** %648, align 4
  %652 = load %struct.edge_rec*, %struct.edge_rec** %632, align 4
  %653 = load %struct.edge_rec*, %struct.edge_rec** %640, align 4
  store %struct.edge_rec* %652, %struct.edge_rec** %640, align 4
  store %struct.edge_rec* %653, %struct.edge_rec** %632, align 4
  %654 = add i32 %630, 48
  %655 = and i32 %654, 63
  %656 = and i32 %598, -64
  %657 = or i32 %655, %656
  %658 = inttoptr i32 %657 to %struct.edge_rec*
  %659 = getelementptr %struct.edge_rec, %struct.edge_rec* %658, i32 0, i32 1
  %660 = load %struct.edge_rec*, %struct.edge_rec** %659, align 4
  %661 = ptrtoint %struct.edge_rec* %660 to i32
  %662 = add i32 %661, 16
  %663 = and i32 %662, 63
  %664 = and i32 %661, -64
  %665 = or i32 %663, %664
  %666 = inttoptr i32 %665 to %struct.edge_rec*
  br label %bb9.i

bb25.i:
  %667 = add i32 %172, 16
  %668 = and i32 %667, 63
  %669 = and i32 %172, -64
  %670 = or i32 %668, %669
  %671 = inttoptr i32 %670 to %struct.edge_rec*
  %672 = getelementptr %struct.edge_rec, %struct.edge_rec* %671, i32 0, i32 1
  %673 = load %struct.edge_rec*, %struct.edge_rec** %672, align 4
  %674 = ptrtoint %struct.edge_rec* %673 to i32
  %675 = add i32 %674, 16
  %676 = and i32 %675, 63
  %677 = and i32 %674, -64
  %678 = or i32 %676, %677
  %679 = inttoptr i32 %678 to %struct.edge_rec*
  %680 = call  %struct.edge_rec* @alloc_edge() nounwind
  %681 = getelementptr %struct.edge_rec, %struct.edge_rec* %680, i32 0, i32 1
  store %struct.edge_rec* %680, %struct.edge_rec** %681, align 4
  %682 = getelementptr %struct.edge_rec, %struct.edge_rec* %680, i32 0, i32 0
  store %struct.VERTEX* %501, %struct.VERTEX** %682, align 4
  %683 = ptrtoint %struct.edge_rec* %680 to i32
  %684 = add i32 %683, 16
  %685 = inttoptr i32 %684 to %struct.edge_rec*
  %686 = add i32 %683, 48
  %687 = inttoptr i32 %686 to %struct.edge_rec*
  %688 = getelementptr %struct.edge_rec, %struct.edge_rec* %685, i32 0, i32 1
  store %struct.edge_rec* %687, %struct.edge_rec** %688, align 4
  %689 = add i32 %683, 32
  %690 = inttoptr i32 %689 to %struct.edge_rec*
  %691 = getelementptr %struct.edge_rec, %struct.edge_rec* %690, i32 0, i32 1
  store %struct.edge_rec* %690, %struct.edge_rec** %691, align 4
  %692 = getelementptr %struct.edge_rec, %struct.edge_rec* %690, i32 0, i32 0
  store %struct.VERTEX* %496, %struct.VERTEX** %692, align 4
  %693 = getelementptr %struct.edge_rec, %struct.edge_rec* %687, i32 0, i32 1
  store %struct.edge_rec* %685, %struct.edge_rec** %693, align 4
  %694 = load %struct.edge_rec*, %struct.edge_rec** %681, align 4
  %695 = ptrtoint %struct.edge_rec* %694 to i32
  %696 = add i32 %695, 16
  %697 = and i32 %696, 63
  %698 = and i32 %695, -64
  %699 = or i32 %697, %698
  %700 = inttoptr i32 %699 to %struct.edge_rec*
  %701 = getelementptr %struct.edge_rec, %struct.edge_rec* %499, i32 0, i32 1
  %702 = load %struct.edge_rec*, %struct.edge_rec** %701, align 4
  %703 = ptrtoint %struct.edge_rec* %702 to i32
  %704 = add i32 %703, 16
  %705 = and i32 %704, 63
  %706 = and i32 %703, -64
  %707 = or i32 %705, %706
  %708 = inttoptr i32 %707 to %struct.edge_rec*
  %709 = getelementptr %struct.edge_rec, %struct.edge_rec* %708, i32 0, i32 1
  %710 = load %struct.edge_rec*, %struct.edge_rec** %709, align 4
  %711 = getelementptr %struct.edge_rec, %struct.edge_rec* %700, i32 0, i32 1
  %712 = load %struct.edge_rec*, %struct.edge_rec** %711, align 4
  store %struct.edge_rec* %710, %struct.edge_rec** %711, align 4
  store %struct.edge_rec* %712, %struct.edge_rec** %709, align 4
  %713 = load %struct.edge_rec*, %struct.edge_rec** %681, align 4
  %714 = load %struct.edge_rec*, %struct.edge_rec** %701, align 4
  store %struct.edge_rec* %713, %struct.edge_rec** %701, align 4
  store %struct.edge_rec* %714, %struct.edge_rec** %681, align 4
  %715 = xor i32 %683, 32
  %716 = inttoptr i32 %715 to %struct.edge_rec*
  %717 = getelementptr %struct.edge_rec, %struct.edge_rec* %716, i32 0, i32 1
  %718 = load %struct.edge_rec*, %struct.edge_rec** %717, align 4
  %719 = ptrtoint %struct.edge_rec* %718 to i32
  %720 = add i32 %719, 16
  %721 = and i32 %720, 63
  %722 = and i32 %719, -64
  %723 = or i32 %721, %722
  %724 = inttoptr i32 %723 to %struct.edge_rec*
  %725 = getelementptr %struct.edge_rec, %struct.edge_rec* %679, i32 0, i32 1
  %726 = load %struct.edge_rec*, %struct.edge_rec** %725, align 4
  %727 = ptrtoint %struct.edge_rec* %726 to i32
  %728 = add i32 %727, 16
  %729 = and i32 %728, 63
  %730 = and i32 %727, -64
  %731 = or i32 %729, %730
  %732 = inttoptr i32 %731 to %struct.edge_rec*
  %733 = getelementptr %struct.edge_rec, %struct.edge_rec* %732, i32 0, i32 1
  %734 = load %struct.edge_rec*, %struct.edge_rec** %733, align 4
  %735 = getelementptr %struct.edge_rec, %struct.edge_rec* %724, i32 0, i32 1
  %736 = load %struct.edge_rec*, %struct.edge_rec** %735, align 4
  store %struct.edge_rec* %734, %struct.edge_rec** %735, align 4
  store %struct.edge_rec* %736, %struct.edge_rec** %733, align 4
  %737 = load %struct.edge_rec*, %struct.edge_rec** %717, align 4
  %738 = load %struct.edge_rec*, %struct.edge_rec** %725, align 4
  store %struct.edge_rec* %737, %struct.edge_rec** %725, align 4
  store %struct.edge_rec* %738, %struct.edge_rec** %717, align 4
  %739 = load %struct.edge_rec*, %struct.edge_rec** %681, align 4
  br label %bb9.i

do_merge.exit:
  %740 = getelementptr %struct.edge_rec, %struct.edge_rec* %ldo_addr.0.ph.i, i32 0, i32 0
  %741 = load %struct.VERTEX*, %struct.VERTEX** %740, align 4
  %742 = icmp eq %struct.VERTEX* %741, %tree_addr.0.i
  br i1 %742, label %bb5.loopexit, label %bb2

bb2:
  %ldo.07 = phi %struct.edge_rec* [ %747, %bb2 ], [ %ldo_addr.0.ph.i, %do_merge.exit ]
  %743 = ptrtoint %struct.edge_rec* %ldo.07 to i32
  %744 = xor i32 %743, 32
  %745 = inttoptr i32 %744 to %struct.edge_rec*
  %746 = getelementptr %struct.edge_rec, %struct.edge_rec* %745, i32 0, i32 1
  %747 = load %struct.edge_rec*, %struct.edge_rec** %746, align 4
  %748 = getelementptr %struct.edge_rec, %struct.edge_rec* %747, i32 0, i32 0
  %749 = load %struct.VERTEX*, %struct.VERTEX** %748, align 4
  %750 = icmp eq %struct.VERTEX* %749, %tree_addr.0.i
  br i1 %750, label %bb5.loopexit, label %bb2

bb4:
  %rdo.05 = phi %struct.edge_rec* [ %755, %bb4 ], [ %rdo_addr.0.i, %bb5.loopexit ]
  %751 = getelementptr %struct.edge_rec, %struct.edge_rec* %rdo.05, i32 0, i32 1
  %752 = load %struct.edge_rec*, %struct.edge_rec** %751, align 4
  %753 = ptrtoint %struct.edge_rec* %752 to i32
  %754 = xor i32 %753, 32
  %755 = inttoptr i32 %754 to %struct.edge_rec*
  %756 = getelementptr %struct.edge_rec, %struct.edge_rec* %755, i32 0, i32 0
  %757 = load %struct.VERTEX*, %struct.VERTEX** %756, align 4
  %758 = icmp eq %struct.VERTEX* %757, %extra
  br i1 %758, label %bb6, label %bb4

bb5.loopexit:
  %ldo.0.lcssa = phi %struct.edge_rec* [ %ldo_addr.0.ph.i, %do_merge.exit ], [ %747, %bb2 ]
  %759 = getelementptr %struct.edge_rec, %struct.edge_rec* %rdo_addr.0.i, i32 0, i32 0
  %760 = load %struct.VERTEX*, %struct.VERTEX** %759, align 4
  %761 = icmp eq %struct.VERTEX* %760, %extra
  br i1 %761, label %bb6, label %bb4

bb6:
  %rdo.0.lcssa = phi %struct.edge_rec* [ %rdo_addr.0.i, %bb5.loopexit ], [ %755, %bb4 ]
  %tmp16 = ptrtoint %struct.edge_rec* %ldo.0.lcssa to i32
  %tmp4 = ptrtoint %struct.edge_rec* %rdo.0.lcssa to i32
  br label %bb15

bb7:
  %762 = getelementptr %struct.VERTEX, %struct.VERTEX* %tree, i32 0, i32 1
  %763 = load %struct.VERTEX*, %struct.VERTEX** %762, align 4
  %764 = icmp eq %struct.VERTEX* %763, null
  %765 = call  %struct.edge_rec* @alloc_edge() nounwind
  %766 = getelementptr %struct.edge_rec, %struct.edge_rec* %765, i32 0, i32 1
  store %struct.edge_rec* %765, %struct.edge_rec** %766, align 4
  %767 = getelementptr %struct.edge_rec, %struct.edge_rec* %765, i32 0, i32 0
  br i1 %764, label %bb10, label %bb11

bb8:
  %768 = call  i32 @puts(i8* getelementptr ([21 x i8], [21 x i8]* @_2E_str7, i32 0, i32 0)) nounwind
  call  void @exit(i32 -1) noreturn nounwind
  unreachable

bb10:
  store %struct.VERTEX* %tree, %struct.VERTEX** %767, align 4
  %769 = ptrtoint %struct.edge_rec* %765 to i32
  %770 = add i32 %769, 16
  %771 = inttoptr i32 %770 to %struct.edge_rec*
  %772 = add i32 %769, 48
  %773 = inttoptr i32 %772 to %struct.edge_rec*
  %774 = getelementptr %struct.edge_rec, %struct.edge_rec* %771, i32 0, i32 1
  store %struct.edge_rec* %773, %struct.edge_rec** %774, align 4
  %775 = add i32 %769, 32
  %776 = inttoptr i32 %775 to %struct.edge_rec*
  %777 = getelementptr %struct.edge_rec, %struct.edge_rec* %776, i32 0, i32 1
  store %struct.edge_rec* %776, %struct.edge_rec** %777, align 4
  %778 = getelementptr %struct.edge_rec, %struct.edge_rec* %776, i32 0, i32 0
  store %struct.VERTEX* %extra, %struct.VERTEX** %778, align 4
  %779 = getelementptr %struct.edge_rec, %struct.edge_rec* %773, i32 0, i32 1
  store %struct.edge_rec* %771, %struct.edge_rec** %779, align 4
  %780 = xor i32 %769, 32
  br label %bb15

bb11:
  store %struct.VERTEX* %763, %struct.VERTEX** %767, align 4
  %781 = ptrtoint %struct.edge_rec* %765 to i32
  %782 = add i32 %781, 16
  %783 = inttoptr i32 %782 to %struct.edge_rec*
  %784 = add i32 %781, 48
  %785 = inttoptr i32 %784 to %struct.edge_rec*
  %786 = getelementptr %struct.edge_rec, %struct.edge_rec* %783, i32 0, i32 1
  store %struct.edge_rec* %785, %struct.edge_rec** %786, align 4
  %787 = add i32 %781, 32
  %788 = inttoptr i32 %787 to %struct.edge_rec*
  %789 = getelementptr %struct.edge_rec, %struct.edge_rec* %788, i32 0, i32 1
  store %struct.edge_rec* %788, %struct.edge_rec** %789, align 4
  %790 = getelementptr %struct.edge_rec, %struct.edge_rec* %788, i32 0, i32 0
  store %struct.VERTEX* %tree, %struct.VERTEX** %790, align 4
  %791 = getelementptr %struct.edge_rec, %struct.edge_rec* %785, i32 0, i32 1
  store %struct.edge_rec* %783, %struct.edge_rec** %791, align 4
  %792 = call  %struct.edge_rec* @alloc_edge() nounwind
  %793 = getelementptr %struct.edge_rec, %struct.edge_rec* %792, i32 0, i32 1
  store %struct.edge_rec* %792, %struct.edge_rec** %793, align 4
  %794 = getelementptr %struct.edge_rec, %struct.edge_rec* %792, i32 0, i32 0
  store %struct.VERTEX* %tree, %struct.VERTEX** %794, align 4
  %795 = ptrtoint %struct.edge_rec* %792 to i32
  %796 = add i32 %795, 16
  %797 = inttoptr i32 %796 to %struct.edge_rec*
  %798 = add i32 %795, 48
  %799 = inttoptr i32 %798 to %struct.edge_rec*
  %800 = getelementptr %struct.edge_rec, %struct.edge_rec* %797, i32 0, i32 1
  store %struct.edge_rec* %799, %struct.edge_rec** %800, align 4
  %801 = add i32 %795, 32
  %802 = inttoptr i32 %801 to %struct.edge_rec*
  %803 = getelementptr %struct.edge_rec, %struct.edge_rec* %802, i32 0, i32 1
  store %struct.edge_rec* %802, %struct.edge_rec** %803, align 4
  %804 = getelementptr %struct.edge_rec, %struct.edge_rec* %802, i32 0, i32 0
  store %struct.VERTEX* %extra, %struct.VERTEX** %804, align 4
  %805 = getelementptr %struct.edge_rec, %struct.edge_rec* %799, i32 0, i32 1
  store %struct.edge_rec* %797, %struct.edge_rec** %805, align 4
  %806 = xor i32 %781, 32
  %807 = inttoptr i32 %806 to %struct.edge_rec*
  %808 = getelementptr %struct.edge_rec, %struct.edge_rec* %807, i32 0, i32 1
  %809 = load %struct.edge_rec*, %struct.edge_rec** %808, align 4
  %810 = ptrtoint %struct.edge_rec* %809 to i32
  %811 = add i32 %810, 16
  %812 = and i32 %811, 63
  %813 = and i32 %810, -64
  %814 = or i32 %812, %813
  %815 = inttoptr i32 %814 to %struct.edge_rec*
  %816 = load %struct.edge_rec*, %struct.edge_rec** %793, align 4
  %817 = ptrtoint %struct.edge_rec* %816 to i32
  %818 = add i32 %817, 16
  %819 = and i32 %818, 63
  %820 = and i32 %817, -64
  %821 = or i32 %819, %820
  %822 = inttoptr i32 %821 to %struct.edge_rec*
  %823 = getelementptr %struct.edge_rec, %struct.edge_rec* %822, i32 0, i32 1
  %824 = load %struct.edge_rec*, %struct.edge_rec** %823, align 4
  %825 = getelementptr %struct.edge_rec, %struct.edge_rec* %815, i32 0, i32 1
  %826 = load %struct.edge_rec*, %struct.edge_rec** %825, align 4
  store %struct.edge_rec* %824, %struct.edge_rec** %825, align 4
  store %struct.edge_rec* %826, %struct.edge_rec** %823, align 4
  %827 = load %struct.edge_rec*, %struct.edge_rec** %808, align 4
  %828 = load %struct.edge_rec*, %struct.edge_rec** %793, align 4
  store %struct.edge_rec* %827, %struct.edge_rec** %793, align 4
  store %struct.edge_rec* %828, %struct.edge_rec** %808, align 4
  %829 = xor i32 %795, 32
  %830 = inttoptr i32 %829 to %struct.edge_rec*
  %831 = getelementptr %struct.edge_rec, %struct.edge_rec* %830, i32 0, i32 0
  %832 = load %struct.VERTEX*, %struct.VERTEX** %831, align 4
  %833 = and i32 %798, 63
  %834 = and i32 %795, -64
  %835 = or i32 %833, %834
  %836 = inttoptr i32 %835 to %struct.edge_rec*
  %837 = getelementptr %struct.edge_rec, %struct.edge_rec* %836, i32 0, i32 1
  %838 = load %struct.edge_rec*, %struct.edge_rec** %837, align 4
  %839 = ptrtoint %struct.edge_rec* %838 to i32
  %840 = add i32 %839, 16
  %841 = and i32 %840, 63
  %842 = and i32 %839, -64
  %843 = or i32 %841, %842
  %844 = inttoptr i32 %843 to %struct.edge_rec*
  %845 = load %struct.VERTEX*, %struct.VERTEX** %767, align 4
  %846 = call  %struct.edge_rec* @alloc_edge() nounwind
  %847 = getelementptr %struct.edge_rec, %struct.edge_rec* %846, i32 0, i32 1
  store %struct.edge_rec* %846, %struct.edge_rec** %847, align 4
  %848 = getelementptr %struct.edge_rec, %struct.edge_rec* %846, i32 0, i32 0
  store %struct.VERTEX* %832, %struct.VERTEX** %848, align 4
  %849 = ptrtoint %struct.edge_rec* %846 to i32
  %850 = add i32 %849, 16
  %851 = inttoptr i32 %850 to %struct.edge_rec*
  %852 = add i32 %849, 48
  %853 = inttoptr i32 %852 to %struct.edge_rec*
  %854 = getelementptr %struct.edge_rec, %struct.edge_rec* %851, i32 0, i32 1
  store %struct.edge_rec* %853, %struct.edge_rec** %854, align 4
  %855 = add i32 %849, 32
  %856 = inttoptr i32 %855 to %struct.edge_rec*
  %857 = getelementptr %struct.edge_rec, %struct.edge_rec* %856, i32 0, i32 1
  store %struct.edge_rec* %856, %struct.edge_rec** %857, align 4
  %858 = getelementptr %struct.edge_rec, %struct.edge_rec* %856, i32 0, i32 0
  store %struct.VERTEX* %845, %struct.VERTEX** %858, align 4
  %859 = getelementptr %struct.edge_rec, %struct.edge_rec* %853, i32 0, i32 1
  store %struct.edge_rec* %851, %struct.edge_rec** %859, align 4
  %860 = load %struct.edge_rec*, %struct.edge_rec** %847, align 4
  %861 = ptrtoint %struct.edge_rec* %860 to i32
  %862 = add i32 %861, 16
  %863 = and i32 %862, 63
  %864 = and i32 %861, -64
  %865 = or i32 %863, %864
  %866 = inttoptr i32 %865 to %struct.edge_rec*
  %867 = getelementptr %struct.edge_rec, %struct.edge_rec* %844, i32 0, i32 1
  %868 = load %struct.edge_rec*, %struct.edge_rec** %867, align 4
  %869 = ptrtoint %struct.edge_rec* %868 to i32
  %870 = add i32 %869, 16
  %871 = and i32 %870, 63
  %872 = and i32 %869, -64
  %873 = or i32 %871, %872
  %874 = inttoptr i32 %873 to %struct.edge_rec*
  %875 = getelementptr %struct.edge_rec, %struct.edge_rec* %874, i32 0, i32 1
  %876 = load %struct.edge_rec*, %struct.edge_rec** %875, align 4
  %877 = getelementptr %struct.edge_rec, %struct.edge_rec* %866, i32 0, i32 1
  %878 = load %struct.edge_rec*, %struct.edge_rec** %877, align 4
  store %struct.edge_rec* %876, %struct.edge_rec** %877, align 4
  store %struct.edge_rec* %878, %struct.edge_rec** %875, align 4
  %879 = load %struct.edge_rec*, %struct.edge_rec** %847, align 4
  %880 = load %struct.edge_rec*, %struct.edge_rec** %867, align 4
  store %struct.edge_rec* %879, %struct.edge_rec** %867, align 4
  store %struct.edge_rec* %880, %struct.edge_rec** %847, align 4
  %881 = xor i32 %849, 32
  %882 = inttoptr i32 %881 to %struct.edge_rec*
  %883 = getelementptr %struct.edge_rec, %struct.edge_rec* %882, i32 0, i32 1
  %884 = load %struct.edge_rec*, %struct.edge_rec** %883, align 4
  %885 = ptrtoint %struct.edge_rec* %884 to i32
  %886 = add i32 %885, 16
  %887 = and i32 %886, 63
  %888 = and i32 %885, -64
  %889 = or i32 %887, %888
  %890 = inttoptr i32 %889 to %struct.edge_rec*
  %891 = load %struct.edge_rec*, %struct.edge_rec** %766, align 4
  %892 = ptrtoint %struct.edge_rec* %891 to i32
  %893 = add i32 %892, 16
  %894 = and i32 %893, 63
  %895 = and i32 %892, -64
  %896 = or i32 %894, %895
  %897 = inttoptr i32 %896 to %struct.edge_rec*
  %898 = getelementptr %struct.edge_rec, %struct.edge_rec* %897, i32 0, i32 1
  %899 = load %struct.edge_rec*, %struct.edge_rec** %898, align 4
  %900 = getelementptr %struct.edge_rec, %struct.edge_rec* %890, i32 0, i32 1
  %901 = load %struct.edge_rec*, %struct.edge_rec** %900, align 4
  store %struct.edge_rec* %899, %struct.edge_rec** %900, align 4
  store %struct.edge_rec* %901, %struct.edge_rec** %898, align 4
  %902 = load %struct.edge_rec*, %struct.edge_rec** %883, align 4
  %903 = load %struct.edge_rec*, %struct.edge_rec** %766, align 4
  store %struct.edge_rec* %902, %struct.edge_rec** %766, align 4
  store %struct.edge_rec* %903, %struct.edge_rec** %883, align 4
  %904 = getelementptr %struct.VERTEX, %struct.VERTEX* %763, i32 0, i32 0, i32 0
  %905 = load double, double* %904, align 4
  %906 = getelementptr %struct.VERTEX, %struct.VERTEX* %763, i32 0, i32 0, i32 1
  %907 = load double, double* %906, align 4
  %908 = getelementptr %struct.VERTEX, %struct.VERTEX* %extra, i32 0, i32 0, i32 0
  %909 = load double, double* %908, align 4
  %910 = getelementptr %struct.VERTEX, %struct.VERTEX* %extra, i32 0, i32 0, i32 1
  %911 = load double, double* %910, align 4
  %912 = getelementptr %struct.VERTEX, %struct.VERTEX* %tree, i32 0, i32 0, i32 0
  %913 = load double, double* %912, align 4
  %914 = getelementptr %struct.VERTEX, %struct.VERTEX* %tree, i32 0, i32 0, i32 1
  %915 = load double, double* %914, align 4
  %916 = fsub double %905, %913
  %917 = fsub double %911, %915
  %918 = fmul double %916, %917
  %919 = fsub double %909, %913
  %920 = fsub double %907, %915
  %921 = fmul double %919, %920
  %922 = fsub double %918, %921
  %923 = fcmp ogt double %922, 0.000000e+00
  br i1 %923, label %bb15, label %bb13

bb13:
  %924 = fsub double %905, %909
  %925 = fsub double %915, %911
  %926 = fmul double %924, %925
  %927 = fsub double %913, %909
  %928 = fsub double %907, %911
  %929 = fmul double %927, %928
  %930 = fsub double %926, %929
  %931 = fcmp ogt double %930, 0.000000e+00
  br i1 %931, label %bb15, label %bb14

bb14:
  %932 = and i32 %850, 63
  %933 = and i32 %849, -64
  %934 = or i32 %932, %933
  %935 = inttoptr i32 %934 to %struct.edge_rec*
  %936 = getelementptr %struct.edge_rec, %struct.edge_rec* %935, i32 0, i32 1
  %937 = load %struct.edge_rec*, %struct.edge_rec** %936, align 4
  %938 = ptrtoint %struct.edge_rec* %937 to i32
  %939 = add i32 %938, 16
  %940 = and i32 %939, 63
  %941 = and i32 %938, -64
  %942 = or i32 %940, %941
  %943 = inttoptr i32 %942 to %struct.edge_rec*
  %944 = load %struct.edge_rec*, %struct.edge_rec** %847, align 4
  %945 = ptrtoint %struct.edge_rec* %944 to i32
  %946 = add i32 %945, 16
  %947 = and i32 %946, 63
  %948 = and i32 %945, -64
  %949 = or i32 %947, %948
  %950 = inttoptr i32 %949 to %struct.edge_rec*
  %951 = getelementptr %struct.edge_rec, %struct.edge_rec* %943, i32 0, i32 1
  %952 = load %struct.edge_rec*, %struct.edge_rec** %951, align 4
  %953 = ptrtoint %struct.edge_rec* %952 to i32
  %954 = add i32 %953, 16
  %955 = and i32 %954, 63
  %956 = and i32 %953, -64
  %957 = or i32 %955, %956
  %958 = inttoptr i32 %957 to %struct.edge_rec*
  %959 = getelementptr %struct.edge_rec, %struct.edge_rec* %958, i32 0, i32 1
  %960 = load %struct.edge_rec*, %struct.edge_rec** %959, align 4
  %961 = getelementptr %struct.edge_rec, %struct.edge_rec* %950, i32 0, i32 1
  %962 = load %struct.edge_rec*, %struct.edge_rec** %961, align 4
  store %struct.edge_rec* %960, %struct.edge_rec** %961, align 4
  store %struct.edge_rec* %962, %struct.edge_rec** %959, align 4
  %963 = load %struct.edge_rec*, %struct.edge_rec** %847, align 4
  %964 = load %struct.edge_rec*, %struct.edge_rec** %951, align 4
  store %struct.edge_rec* %963, %struct.edge_rec** %951, align 4
  store %struct.edge_rec* %964, %struct.edge_rec** %847, align 4
  %965 = add i32 %881, 16
  %966 = and i32 %965, 63
  %967 = or i32 %966, %933
  %968 = inttoptr i32 %967 to %struct.edge_rec*
  %969 = getelementptr %struct.edge_rec, %struct.edge_rec* %968, i32 0, i32 1
  %970 = load %struct.edge_rec*, %struct.edge_rec** %969, align 4
  %971 = ptrtoint %struct.edge_rec* %970 to i32
  %972 = add i32 %971, 16
  %973 = and i32 %972, 63
  %974 = and i32 %971, -64
  %975 = or i32 %973, %974
  %976 = inttoptr i32 %975 to %struct.edge_rec*
  %977 = load %struct.edge_rec*, %struct.edge_rec** %883, align 4
  %978 = ptrtoint %struct.edge_rec* %977 to i32
  %979 = add i32 %978, 16
  %980 = and i32 %979, 63
  %981 = and i32 %978, -64
  %982 = or i32 %980, %981
  %983 = inttoptr i32 %982 to %struct.edge_rec*
  %984 = getelementptr %struct.edge_rec, %struct.edge_rec* %976, i32 0, i32 1
  %985 = load %struct.edge_rec*, %struct.edge_rec** %984, align 4
  %986 = ptrtoint %struct.edge_rec* %985 to i32
  %987 = add i32 %986, 16
  %988 = and i32 %987, 63
  %989 = and i32 %986, -64
  %990 = or i32 %988, %989
  %991 = inttoptr i32 %990 to %struct.edge_rec*
  %992 = getelementptr %struct.edge_rec, %struct.edge_rec* %991, i32 0, i32 1
  %993 = load %struct.edge_rec*, %struct.edge_rec** %992, align 4
  %994 = getelementptr %struct.edge_rec, %struct.edge_rec* %983, i32 0, i32 1
  %995 = load %struct.edge_rec*, %struct.edge_rec** %994, align 4
  store %struct.edge_rec* %993, %struct.edge_rec** %994, align 4
  store %struct.edge_rec* %995, %struct.edge_rec** %992, align 4
  %996 = load %struct.edge_rec*, %struct.edge_rec** %883, align 4
  %997 = load %struct.edge_rec*, %struct.edge_rec** %984, align 4
  store %struct.edge_rec* %996, %struct.edge_rec** %984, align 4
  store %struct.edge_rec* %997, %struct.edge_rec** %883, align 4
  %998 = inttoptr i32 %933 to %struct.edge_rec*
  %999 = load %struct.edge_rec*, %struct.edge_rec** @avail_edge, align 4
  %1000 = getelementptr %struct.edge_rec, %struct.edge_rec* %998, i32 0, i32 1
  store %struct.edge_rec* %999, %struct.edge_rec** %1000, align 4
  store %struct.edge_rec* %998, %struct.edge_rec** @avail_edge, align 4
  br label %bb15

bb15:
  %retval.1.0 = phi i32 [ %780, %bb10 ], [ %829, %bb13 ], [ %829, %bb14 ], [ %tmp4, %bb6 ], [ %849, %bb11 ]
  %retval.0.0 = phi i32 [ %769, %bb10 ], [ %781, %bb13 ], [ %781, %bb14 ], [ %tmp16, %bb6 ], [ %881, %bb11 ]
  %agg.result162 = bitcast %struct.EDGE_PAIR* %agg.result to i64*
  %1001 = zext i32 %retval.0.0 to i64
  %1002 = zext i32 %retval.1.0 to i64
  %1003 = shl i64 %1002, 32
  %1004 = or i64 %1003, %1001
  store i64 %1004, i64* %agg.result162, align 4
  ret void
}

; CHECK-LABEL: _build_delaunay:
; CHECK: vcmp
; CHECK: vcmp
; CHECK: vcmp
; CHECK: vcmp
; CHECK: vcmp
; CHECK: vcmp
; CHECK: vcmp
; CHECK: vcmp
; CHECK: vcmp
; CHECK: vcmp
; CHECK: vcmp
; CHECK: vcmp
; CHECK: vcmp

declare i32 @puts(i8* nocapture) nounwind

declare void @exit(i32) noreturn nounwind

declare %struct.edge_rec* @alloc_edge() nounwind
