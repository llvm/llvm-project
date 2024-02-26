; RUN: llc -mtriple armv6-apple-darwin10 -mattr=+vfp2 -filetype asm -o - %s | FileCheck %s

%struct.EDGE_PAIR = type { ptr, ptr }
%struct.VEC2 = type { double, double, double }
%struct.VERTEX = type { %struct.VEC2, ptr, ptr }
%struct.edge_rec = type { ptr, ptr, i32, ptr }
@avail_edge = internal global ptr null
@_2E_str7 = internal constant [21 x i8] c"ERROR: Only 1 point!\00", section "__TEXT,__cstring,cstring_literals", align 1
@llvm.used = appending global [1 x ptr] [ptr @build_delaunay], section "llvm.metadata"

define void @build_delaunay(ptr noalias nocapture sret(%struct.EDGE_PAIR) %agg.result, ptr %tree, ptr %extra) nounwind {
entry:
  %delright = alloca %struct.EDGE_PAIR, align 8
  %delleft = alloca %struct.EDGE_PAIR, align 8
  %0 = icmp eq ptr %tree, null
  br i1 %0, label %bb8, label %bb

bb:
  %1 = getelementptr %struct.VERTEX, ptr %tree, i32 0, i32 2
  %2 = load ptr, ptr %1, align 4
  %3 = icmp eq ptr %2, null
  br i1 %3, label %bb7, label %bb1.i

bb1.i:
  %tree_addr.0.i = phi ptr [ %5, %bb1.i ], [ %tree, %bb ]
  %4 = getelementptr %struct.VERTEX, ptr %tree_addr.0.i, i32 0, i32 1
  %5 = load ptr, ptr %4, align 4
  %6 = icmp eq ptr %5, null
  br i1 %6, label %get_low.exit, label %bb1.i

get_low.exit:
  call  void @build_delaunay(ptr noalias sret(%struct.EDGE_PAIR) %delright, ptr %2, ptr %extra) nounwind
  %7 = getelementptr %struct.VERTEX, ptr %tree, i32 0, i32 1
  %8 = load ptr, ptr %7, align 4
  call  void @build_delaunay(ptr noalias sret(%struct.EDGE_PAIR) %delleft, ptr %8, ptr %tree) nounwind
  %9 = getelementptr %struct.EDGE_PAIR, ptr %delleft, i32 0, i32 0
  %10 = load ptr, ptr %9, align 8
  %11 = getelementptr %struct.EDGE_PAIR, ptr %delleft, i32 0, i32 1
  %12 = load ptr, ptr %11, align 4
  %13 = getelementptr %struct.EDGE_PAIR, ptr %delright, i32 0, i32 0
  %14 = load ptr, ptr %13, align 8
  %15 = getelementptr %struct.EDGE_PAIR, ptr %delright, i32 0, i32 1
  %16 = load ptr, ptr %15, align 4
  br label %bb.i

bb.i:
  %rdi_addr.0.i = phi ptr [ %14, %get_low.exit ], [ %72, %bb4.i ]
  %ldi_addr.1.i = phi ptr [ %12, %get_low.exit ], [ %ldi_addr.0.i, %bb4.i ]
  %17 = getelementptr %struct.edge_rec, ptr %rdi_addr.0.i, i32 0, i32 0
  %18 = load ptr, ptr %17, align 4
  %19 = ptrtoint ptr %ldi_addr.1.i to i32
  %20 = getelementptr %struct.VERTEX, ptr %18, i32 0, i32 0, i32 0
  %21 = load double, ptr %20, align 4
  %22 = getelementptr %struct.VERTEX, ptr %18, i32 0, i32 0, i32 1
  %23 = load double, ptr %22, align 4
  br label %bb2.i

bb1.i1:
  %24 = ptrtoint ptr %ldi_addr.0.i to i32
  %25 = add i32 %24, 48
  %26 = and i32 %25, 63
  %27 = and i32 %24, -64
  %28 = or i32 %26, %27
  %29 = inttoptr i32 %28 to ptr
  %30 = getelementptr %struct.edge_rec, ptr %29, i32 0, i32 1
  %31 = load ptr, ptr %30, align 4
  %32 = ptrtoint ptr %31 to i32
  %33 = add i32 %32, 16
  %34 = and i32 %33, 63
  %35 = and i32 %32, -64
  %36 = or i32 %34, %35
  %37 = inttoptr i32 %36 to ptr
  br label %bb2.i

bb2.i:
  %ldi_addr.1.pn.i = phi ptr [ %ldi_addr.1.i, %bb.i ], [ %37, %bb1.i1 ]
  %.pn6.in.in.i = phi i32 [ %19, %bb.i ], [ %36, %bb1.i1 ]
  %ldi_addr.0.i = phi ptr [ %ldi_addr.1.i, %bb.i ], [ %37, %bb1.i1 ]
  %.pn6.in.i = xor i32 %.pn6.in.in.i, 32
  %.pn6.i = inttoptr i32 %.pn6.in.i to ptr
  %t1.0.in.i = getelementptr %struct.edge_rec, ptr %ldi_addr.1.pn.i, i32 0, i32 0
  %t2.0.in.i = getelementptr %struct.edge_rec, ptr %.pn6.i, i32 0, i32 0
  %t1.0.i = load ptr, ptr %t1.0.in.i
  %t2.0.i = load ptr, ptr %t2.0.in.i
  %38 = getelementptr %struct.VERTEX, ptr %t1.0.i, i32 0, i32 0, i32 0
  %39 = load double, ptr %38, align 4
  %40 = getelementptr %struct.VERTEX, ptr %t1.0.i, i32 0, i32 0, i32 1
  %41 = load double, ptr %40, align 4
  %42 = getelementptr %struct.VERTEX, ptr %t2.0.i, i32 0, i32 0, i32 0
  %43 = load double, ptr %42, align 4
  %44 = getelementptr %struct.VERTEX, ptr %t2.0.i, i32 0, i32 0, i32 1
  %45 = load double, ptr %44, align 4
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
  %54 = ptrtoint ptr %rdi_addr.0.i to i32
  %55 = xor i32 %54, 32
  %56 = inttoptr i32 %55 to ptr
  %57 = getelementptr %struct.edge_rec, ptr %56, i32 0, i32 0
  %58 = load ptr, ptr %57, align 4
  %59 = getelementptr %struct.VERTEX, ptr %58, i32 0, i32 0, i32 0
  %60 = load double, ptr %59, align 4
  %61 = getelementptr %struct.VERTEX, ptr %58, i32 0, i32 0, i32 1
  %62 = load double, ptr %61, align 4
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
  %71 = getelementptr %struct.edge_rec, ptr %56, i32 0, i32 1
  %72 = load ptr, ptr %71, align 4
  br label %bb.i

bb5.i:
  %73 = add i32 %55, 48
  %74 = and i32 %73, 63
  %75 = and i32 %55, -64
  %76 = or i32 %74, %75
  %77 = inttoptr i32 %76 to ptr
  %78 = getelementptr %struct.edge_rec, ptr %77, i32 0, i32 1
  %79 = load ptr, ptr %78, align 4
  %80 = ptrtoint ptr %79 to i32
  %81 = add i32 %80, 16
  %82 = and i32 %81, 63
  %83 = and i32 %80, -64
  %84 = or i32 %82, %83
  %85 = inttoptr i32 %84 to ptr
  %86 = getelementptr %struct.edge_rec, ptr %ldi_addr.0.i, i32 0, i32 0
  %87 = load ptr, ptr %86, align 4
  %88 = call  ptr @alloc_edge() nounwind
  %89 = getelementptr %struct.edge_rec, ptr %88, i32 0, i32 1
  store ptr %88, ptr %89, align 4
  %90 = getelementptr %struct.edge_rec, ptr %88, i32 0, i32 0
  store ptr %18, ptr %90, align 4
  %91 = ptrtoint ptr %88 to i32
  %92 = add i32 %91, 16
  %93 = inttoptr i32 %92 to ptr
  %94 = add i32 %91, 48
  %95 = inttoptr i32 %94 to ptr
  %96 = getelementptr %struct.edge_rec, ptr %93, i32 0, i32 1
  store ptr %95, ptr %96, align 4
  %97 = add i32 %91, 32
  %98 = inttoptr i32 %97 to ptr
  %99 = getelementptr %struct.edge_rec, ptr %98, i32 0, i32 1
  store ptr %98, ptr %99, align 4
  %100 = getelementptr %struct.edge_rec, ptr %98, i32 0, i32 0
  store ptr %87, ptr %100, align 4
  %101 = getelementptr %struct.edge_rec, ptr %95, i32 0, i32 1
  store ptr %93, ptr %101, align 4
  %102 = load ptr, ptr %89, align 4
  %103 = ptrtoint ptr %102 to i32
  %104 = add i32 %103, 16
  %105 = and i32 %104, 63
  %106 = and i32 %103, -64
  %107 = or i32 %105, %106
  %108 = inttoptr i32 %107 to ptr
  %109 = getelementptr %struct.edge_rec, ptr %85, i32 0, i32 1
  %110 = load ptr, ptr %109, align 4
  %111 = ptrtoint ptr %110 to i32
  %112 = add i32 %111, 16
  %113 = and i32 %112, 63
  %114 = and i32 %111, -64
  %115 = or i32 %113, %114
  %116 = inttoptr i32 %115 to ptr
  %117 = getelementptr %struct.edge_rec, ptr %116, i32 0, i32 1
  %118 = load ptr, ptr %117, align 4
  %119 = getelementptr %struct.edge_rec, ptr %108, i32 0, i32 1
  %120 = load ptr, ptr %119, align 4
  store ptr %118, ptr %119, align 4
  store ptr %120, ptr %117, align 4
  %121 = load ptr, ptr %89, align 4
  %122 = load ptr, ptr %109, align 4
  store ptr %121, ptr %109, align 4
  store ptr %122, ptr %89, align 4
  %123 = xor i32 %91, 32
  %124 = inttoptr i32 %123 to ptr
  %125 = getelementptr %struct.edge_rec, ptr %124, i32 0, i32 1
  %126 = load ptr, ptr %125, align 4
  %127 = ptrtoint ptr %126 to i32
  %128 = add i32 %127, 16
  %129 = and i32 %128, 63
  %130 = and i32 %127, -64
  %131 = or i32 %129, %130
  %132 = inttoptr i32 %131 to ptr
  %133 = getelementptr %struct.edge_rec, ptr %ldi_addr.0.i, i32 0, i32 1
  %134 = load ptr, ptr %133, align 4
  %135 = ptrtoint ptr %134 to i32
  %136 = add i32 %135, 16
  %137 = and i32 %136, 63
  %138 = and i32 %135, -64
  %139 = or i32 %137, %138
  %140 = inttoptr i32 %139 to ptr
  %141 = getelementptr %struct.edge_rec, ptr %140, i32 0, i32 1
  %142 = load ptr, ptr %141, align 4
  %143 = getelementptr %struct.edge_rec, ptr %132, i32 0, i32 1
  %144 = load ptr, ptr %143, align 4
  store ptr %142, ptr %143, align 4
  store ptr %144, ptr %141, align 4
  %145 = load ptr, ptr %125, align 4
  %146 = load ptr, ptr %133, align 4
  store ptr %145, ptr %133, align 4
  store ptr %146, ptr %125, align 4
  %147 = and i32 %92, 63
  %148 = and i32 %91, -64
  %149 = or i32 %147, %148
  %150 = inttoptr i32 %149 to ptr
  %151 = getelementptr %struct.edge_rec, ptr %150, i32 0, i32 1
  %152 = load ptr, ptr %151, align 4
  %153 = ptrtoint ptr %152 to i32
  %154 = add i32 %153, 16
  %155 = and i32 %154, 63
  %156 = and i32 %153, -64
  %157 = or i32 %155, %156
  %158 = inttoptr i32 %157 to ptr
  %159 = load ptr, ptr %90, align 4
  %160 = getelementptr %struct.edge_rec, ptr %124, i32 0, i32 0
  %161 = load ptr, ptr %160, align 4
  %162 = getelementptr %struct.edge_rec, ptr %16, i32 0, i32 0
  %163 = load ptr, ptr %162, align 4
  %164 = icmp eq ptr %163, %159
  %rdo_addr.0.i = select i1 %164, ptr %88, ptr %16
  %165 = getelementptr %struct.edge_rec, ptr %10, i32 0, i32 0
  %166 = load ptr, ptr %165, align 4
  %167 = icmp eq ptr %166, %161
  %ldo_addr.0.ph.i = select i1 %167, ptr %124, ptr %10
  br label %bb9.i

bb9.i:
  %lcand.2.i = phi ptr [ %146, %bb5.i ], [ %lcand.1.i, %bb24.i ], [ %739, %bb25.i ]
  %rcand.2.i = phi ptr [ %158, %bb5.i ], [ %666, %bb24.i ], [ %rcand.1.i, %bb25.i ]
  %basel.0.i = phi ptr [ %88, %bb5.i ], [ %595, %bb24.i ], [ %716, %bb25.i ]
  %168 = getelementptr %struct.edge_rec, ptr %lcand.2.i, i32 0, i32 1
  %169 = load ptr, ptr %168, align 4
  %170 = getelementptr %struct.edge_rec, ptr %basel.0.i, i32 0, i32 0
  %171 = load ptr, ptr %170, align 4
  %172 = ptrtoint ptr %basel.0.i to i32
  %173 = xor i32 %172, 32
  %174 = inttoptr i32 %173 to ptr
  %175 = getelementptr %struct.edge_rec, ptr %174, i32 0, i32 0
  %176 = load ptr, ptr %175, align 4
  %177 = ptrtoint ptr %169 to i32
  %178 = xor i32 %177, 32
  %179 = inttoptr i32 %178 to ptr
  %180 = getelementptr %struct.edge_rec, ptr %179, i32 0, i32 0
  %181 = load ptr, ptr %180, align 4
  %182 = getelementptr %struct.VERTEX, ptr %171, i32 0, i32 0, i32 0
  %183 = load double, ptr %182, align 4
  %184 = getelementptr %struct.VERTEX, ptr %171, i32 0, i32 0, i32 1
  %185 = load double, ptr %184, align 4
  %186 = getelementptr %struct.VERTEX, ptr %181, i32 0, i32 0, i32 0
  %187 = load double, ptr %186, align 4
  %188 = getelementptr %struct.VERTEX, ptr %181, i32 0, i32 0, i32 1
  %189 = load double, ptr %188, align 4
  %190 = getelementptr %struct.VERTEX, ptr %176, i32 0, i32 0, i32 0
  %191 = load double, ptr %190, align 4
  %192 = getelementptr %struct.VERTEX, ptr %176, i32 0, i32 0, i32 1
  %193 = load double, ptr %192, align 4
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
  %202 = getelementptr %struct.VERTEX, ptr %171, i32 0, i32 0, i32 2
  %avail_edge.promoted25 = load ptr, ptr @avail_edge
  br label %bb12.i

bb11.i:
  %203 = ptrtoint ptr %lcand.0.i to i32
  %204 = add i32 %203, 16
  %205 = and i32 %204, 63
  %206 = and i32 %203, -64
  %207 = or i32 %205, %206
  %208 = inttoptr i32 %207 to ptr
  %209 = getelementptr %struct.edge_rec, ptr %208, i32 0, i32 1
  %210 = load ptr, ptr %209, align 4
  %211 = ptrtoint ptr %210 to i32
  %212 = add i32 %211, 16
  %213 = and i32 %212, 63
  %214 = and i32 %211, -64
  %215 = or i32 %213, %214
  %216 = inttoptr i32 %215 to ptr
  %217 = getelementptr %struct.edge_rec, ptr %lcand.0.i, i32 0, i32 1
  %218 = load ptr, ptr %217, align 4
  %219 = ptrtoint ptr %218 to i32
  %220 = add i32 %219, 16
  %221 = and i32 %220, 63
  %222 = and i32 %219, -64
  %223 = or i32 %221, %222
  %224 = inttoptr i32 %223 to ptr
  %225 = getelementptr %struct.edge_rec, ptr %216, i32 0, i32 1
  %226 = load ptr, ptr %225, align 4
  %227 = ptrtoint ptr %226 to i32
  %228 = add i32 %227, 16
  %229 = and i32 %228, 63
  %230 = and i32 %227, -64
  %231 = or i32 %229, %230
  %232 = inttoptr i32 %231 to ptr
  %233 = getelementptr %struct.edge_rec, ptr %232, i32 0, i32 1
  %234 = load ptr, ptr %233, align 4
  %235 = getelementptr %struct.edge_rec, ptr %224, i32 0, i32 1
  %236 = load ptr, ptr %235, align 4
  store ptr %234, ptr %235, align 4
  store ptr %236, ptr %233, align 4
  %237 = load ptr, ptr %217, align 4
  %238 = load ptr, ptr %225, align 4
  store ptr %237, ptr %225, align 4
  store ptr %238, ptr %217, align 4
  %239 = xor i32 %203, 32
  %240 = add i32 %239, 16
  %241 = and i32 %240, 63
  %242 = or i32 %241, %206
  %243 = inttoptr i32 %242 to ptr
  %244 = getelementptr %struct.edge_rec, ptr %243, i32 0, i32 1
  %245 = load ptr, ptr %244, align 4
  %246 = ptrtoint ptr %245 to i32
  %247 = add i32 %246, 16
  %248 = and i32 %247, 63
  %249 = and i32 %246, -64
  %250 = or i32 %248, %249
  %251 = inttoptr i32 %250 to ptr
  %252 = inttoptr i32 %239 to ptr
  %253 = getelementptr %struct.edge_rec, ptr %252, i32 0, i32 1
  %254 = load ptr, ptr %253, align 4
  %255 = ptrtoint ptr %254 to i32
  %256 = add i32 %255, 16
  %257 = and i32 %256, 63
  %258 = and i32 %255, -64
  %259 = or i32 %257, %258
  %260 = inttoptr i32 %259 to ptr
  %261 = getelementptr %struct.edge_rec, ptr %251, i32 0, i32 1
  %262 = load ptr, ptr %261, align 4
  %263 = ptrtoint ptr %262 to i32
  %264 = add i32 %263, 16
  %265 = and i32 %264, 63
  %266 = and i32 %263, -64
  %267 = or i32 %265, %266
  %268 = inttoptr i32 %267 to ptr
  %269 = getelementptr %struct.edge_rec, ptr %268, i32 0, i32 1
  %270 = load ptr, ptr %269, align 4
  %271 = getelementptr %struct.edge_rec, ptr %260, i32 0, i32 1
  %272 = load ptr, ptr %271, align 4
  store ptr %270, ptr %271, align 4
  store ptr %272, ptr %269, align 4
  %273 = load ptr, ptr %253, align 4
  %274 = load ptr, ptr %261, align 4
  store ptr %273, ptr %261, align 4
  store ptr %274, ptr %253, align 4
  %275 = inttoptr i32 %206 to ptr
  %276 = getelementptr %struct.edge_rec, ptr %275, i32 0, i32 1
  store ptr %avail_edge.tmp.026, ptr %276, align 4
  %277 = getelementptr %struct.edge_rec, ptr %t.0.i, i32 0, i32 1
  %278 = load ptr, ptr %277, align 4
  %.pre.i = load double, ptr %182, align 4
  %.pre22.i = load double, ptr %184, align 4
  br label %bb12.i

bb12.i:
  %avail_edge.tmp.026 = phi ptr [ %avail_edge.promoted25, %bb10.i ], [ %275, %bb11.i ]
  %279 = phi double [ %.pre22.i, %bb11.i ], [ %185, %bb10.i ]
  %280 = phi double [ %.pre.i, %bb11.i ], [ %183, %bb10.i ]
  %lcand.0.i = phi ptr [ %lcand.2.i, %bb10.i ], [ %t.0.i, %bb11.i ]
  %t.0.i = phi ptr [ %169, %bb10.i ], [ %278, %bb11.i ]
  %.pn5.in.in.in.i = phi ptr [ %lcand.2.i, %bb10.i ], [ %t.0.i, %bb11.i ]
  %.pn4.in.in.in.i = phi ptr [ %169, %bb10.i ], [ %278, %bb11.i ]
  %lcand.2.pn.i = phi ptr [ %lcand.2.i, %bb10.i ], [ %t.0.i, %bb11.i ]
  %.pn5.in.in.i = ptrtoint ptr %.pn5.in.in.in.i to i32
  %.pn4.in.in.i = ptrtoint ptr %.pn4.in.in.in.i to i32
  %.pn5.in.i = xor i32 %.pn5.in.in.i, 32
  %.pn4.in.i = xor i32 %.pn4.in.in.i, 32
  %.pn5.i = inttoptr i32 %.pn5.in.i to ptr
  %.pn4.i = inttoptr i32 %.pn4.in.i to ptr
  %v1.0.in.i = getelementptr %struct.edge_rec, ptr %.pn5.i, i32 0, i32 0
  %v2.0.in.i = getelementptr %struct.edge_rec, ptr %.pn4.i, i32 0, i32 0
  %v3.0.in.i = getelementptr %struct.edge_rec, ptr %lcand.2.pn.i, i32 0, i32 0
  %v1.0.i = load ptr, ptr %v1.0.in.i
  %v2.0.i = load ptr, ptr %v2.0.in.i
  %v3.0.i = load ptr, ptr %v3.0.in.i
  %281 = load double, ptr %202, align 4
  %282 = getelementptr %struct.VERTEX, ptr %v1.0.i, i32 0, i32 0, i32 0
  %283 = load double, ptr %282, align 4
  %284 = fsub double %283, %280
  %285 = getelementptr %struct.VERTEX, ptr %v1.0.i, i32 0, i32 0, i32 1
  %286 = load double, ptr %285, align 4
  %287 = fsub double %286, %279
  %288 = getelementptr %struct.VERTEX, ptr %v1.0.i, i32 0, i32 0, i32 2
  %289 = load double, ptr %288, align 4
  %290 = getelementptr %struct.VERTEX, ptr %v2.0.i, i32 0, i32 0, i32 0
  %291 = load double, ptr %290, align 4
  %292 = fsub double %291, %280
  %293 = getelementptr %struct.VERTEX, ptr %v2.0.i, i32 0, i32 0, i32 1
  %294 = load double, ptr %293, align 4
  %295 = fsub double %294, %279
  %296 = getelementptr %struct.VERTEX, ptr %v2.0.i, i32 0, i32 0, i32 2
  %297 = load double, ptr %296, align 4
  %298 = getelementptr %struct.VERTEX, ptr %v3.0.i, i32 0, i32 0, i32 0
  %299 = load double, ptr %298, align 4
  %300 = fsub double %299, %280
  %301 = getelementptr %struct.VERTEX, ptr %v3.0.i, i32 0, i32 0, i32 1
  %302 = load double, ptr %301, align 4
  %303 = fsub double %302, %279
  %304 = getelementptr %struct.VERTEX, ptr %v3.0.i, i32 0, i32 0, i32 2
  %305 = load double, ptr %304, align 4
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
  store ptr %avail_edge.tmp.026, ptr @avail_edge
  %.pre23.i = load ptr, ptr %170, align 4
  %.pre24.i = load ptr, ptr %175, align 4
  br label %bb13.i

bb13.i:
  %324 = phi ptr [ %.pre24.i, %bb13.loopexit.i ], [ %176, %bb9.i ]
  %325 = phi ptr [ %.pre23.i, %bb13.loopexit.i ], [ %171, %bb9.i ]
  %lcand.1.i = phi ptr [ %lcand.0.i, %bb13.loopexit.i ], [ %lcand.2.i, %bb9.i ]
  %326 = ptrtoint ptr %rcand.2.i to i32
  %327 = add i32 %326, 16
  %328 = and i32 %327, 63
  %329 = and i32 %326, -64
  %330 = or i32 %328, %329
  %331 = inttoptr i32 %330 to ptr
  %332 = getelementptr %struct.edge_rec, ptr %331, i32 0, i32 1
  %333 = load ptr, ptr %332, align 4
  %334 = ptrtoint ptr %333 to i32
  %335 = add i32 %334, 16
  %336 = and i32 %335, 63
  %337 = and i32 %334, -64
  %338 = or i32 %336, %337
  %339 = xor i32 %338, 32
  %340 = inttoptr i32 %339 to ptr
  %341 = getelementptr %struct.edge_rec, ptr %340, i32 0, i32 0
  %342 = load ptr, ptr %341, align 4
  %343 = getelementptr %struct.VERTEX, ptr %325, i32 0, i32 0, i32 0
  %344 = load double, ptr %343, align 4
  %345 = getelementptr %struct.VERTEX, ptr %325, i32 0, i32 0, i32 1
  %346 = load double, ptr %345, align 4
  %347 = getelementptr %struct.VERTEX, ptr %342, i32 0, i32 0, i32 0
  %348 = load double, ptr %347, align 4
  %349 = getelementptr %struct.VERTEX, ptr %342, i32 0, i32 0, i32 1
  %350 = load double, ptr %349, align 4
  %351 = getelementptr %struct.VERTEX, ptr %324, i32 0, i32 0, i32 0
  %352 = load double, ptr %351, align 4
  %353 = getelementptr %struct.VERTEX, ptr %324, i32 0, i32 0, i32 1
  %354 = load double, ptr %353, align 4
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
  %363 = getelementptr %struct.VERTEX, ptr %324, i32 0, i32 0, i32 2
  %avail_edge.promoted = load ptr, ptr @avail_edge
  br label %bb16.i

bb15.i:
  %364 = ptrtoint ptr %rcand.0.i to i32
  %365 = add i32 %364, 16
  %366 = and i32 %365, 63
  %367 = and i32 %364, -64
  %368 = or i32 %366, %367
  %369 = inttoptr i32 %368 to ptr
  %370 = getelementptr %struct.edge_rec, ptr %369, i32 0, i32 1
  %371 = load ptr, ptr %370, align 4
  %372 = ptrtoint ptr %371 to i32
  %373 = add i32 %372, 16
  %374 = and i32 %373, 63
  %375 = and i32 %372, -64
  %376 = or i32 %374, %375
  %377 = inttoptr i32 %376 to ptr
  %378 = getelementptr %struct.edge_rec, ptr %rcand.0.i, i32 0, i32 1
  %379 = load ptr, ptr %378, align 4
  %380 = ptrtoint ptr %379 to i32
  %381 = add i32 %380, 16
  %382 = and i32 %381, 63
  %383 = and i32 %380, -64
  %384 = or i32 %382, %383
  %385 = inttoptr i32 %384 to ptr
  %386 = getelementptr %struct.edge_rec, ptr %377, i32 0, i32 1
  %387 = load ptr, ptr %386, align 4
  %388 = ptrtoint ptr %387 to i32
  %389 = add i32 %388, 16
  %390 = and i32 %389, 63
  %391 = and i32 %388, -64
  %392 = or i32 %390, %391
  %393 = inttoptr i32 %392 to ptr
  %394 = getelementptr %struct.edge_rec, ptr %393, i32 0, i32 1
  %395 = load ptr, ptr %394, align 4
  %396 = getelementptr %struct.edge_rec, ptr %385, i32 0, i32 1
  %397 = load ptr, ptr %396, align 4
  store ptr %395, ptr %396, align 4
  store ptr %397, ptr %394, align 4
  %398 = load ptr, ptr %378, align 4
  %399 = load ptr, ptr %386, align 4
  store ptr %398, ptr %386, align 4
  store ptr %399, ptr %378, align 4
  %400 = xor i32 %364, 32
  %401 = add i32 %400, 16
  %402 = and i32 %401, 63
  %403 = or i32 %402, %367
  %404 = inttoptr i32 %403 to ptr
  %405 = getelementptr %struct.edge_rec, ptr %404, i32 0, i32 1
  %406 = load ptr, ptr %405, align 4
  %407 = ptrtoint ptr %406 to i32
  %408 = add i32 %407, 16
  %409 = and i32 %408, 63
  %410 = and i32 %407, -64
  %411 = or i32 %409, %410
  %412 = inttoptr i32 %411 to ptr
  %413 = inttoptr i32 %400 to ptr
  %414 = getelementptr %struct.edge_rec, ptr %413, i32 0, i32 1
  %415 = load ptr, ptr %414, align 4
  %416 = ptrtoint ptr %415 to i32
  %417 = add i32 %416, 16
  %418 = and i32 %417, 63
  %419 = and i32 %416, -64
  %420 = or i32 %418, %419
  %421 = inttoptr i32 %420 to ptr
  %422 = getelementptr %struct.edge_rec, ptr %412, i32 0, i32 1
  %423 = load ptr, ptr %422, align 4
  %424 = ptrtoint ptr %423 to i32
  %425 = add i32 %424, 16
  %426 = and i32 %425, 63
  %427 = and i32 %424, -64
  %428 = or i32 %426, %427
  %429 = inttoptr i32 %428 to ptr
  %430 = getelementptr %struct.edge_rec, ptr %429, i32 0, i32 1
  %431 = load ptr, ptr %430, align 4
  %432 = getelementptr %struct.edge_rec, ptr %421, i32 0, i32 1
  %433 = load ptr, ptr %432, align 4
  store ptr %431, ptr %432, align 4
  store ptr %433, ptr %430, align 4
  %434 = load ptr, ptr %414, align 4
  %435 = load ptr, ptr %422, align 4
  store ptr %434, ptr %422, align 4
  store ptr %435, ptr %414, align 4
  %436 = inttoptr i32 %367 to ptr
  %437 = getelementptr %struct.edge_rec, ptr %436, i32 0, i32 1
  store ptr %avail_edge.tmp.0, ptr %437, align 4
  %438 = add i32 %t.1.in.i, 16
  %439 = and i32 %438, 63
  %440 = and i32 %t.1.in.i, -64
  %441 = or i32 %439, %440
  %442 = inttoptr i32 %441 to ptr
  %443 = getelementptr %struct.edge_rec, ptr %442, i32 0, i32 1
  %444 = load ptr, ptr %443, align 4
  %445 = ptrtoint ptr %444 to i32
  %446 = add i32 %445, 16
  %447 = and i32 %446, 63
  %448 = and i32 %445, -64
  %449 = or i32 %447, %448
  %.pre25.i = load double, ptr %351, align 4
  %.pre26.i = load double, ptr %353, align 4
  br label %bb16.i

bb16.i:
  %avail_edge.tmp.0 = phi ptr [ %avail_edge.promoted, %bb14.i ], [ %436, %bb15.i ]
  %450 = phi double [ %.pre26.i, %bb15.i ], [ %354, %bb14.i ]
  %451 = phi double [ %.pre25.i, %bb15.i ], [ %352, %bb14.i ]
  %rcand.0.i = phi ptr [ %rcand.2.i, %bb14.i ], [ %t.1.i, %bb15.i ]
  %t.1.in.i = phi i32 [ %338, %bb14.i ], [ %449, %bb15.i ]
  %.pn3.in.in.i = phi i32 [ %338, %bb14.i ], [ %449, %bb15.i ]
  %.pn.in.in.in.i = phi ptr [ %rcand.2.i, %bb14.i ], [ %t.1.i, %bb15.i ]
  %rcand.2.pn.i = phi ptr [ %rcand.2.i, %bb14.i ], [ %t.1.i, %bb15.i ]
  %t.1.i = inttoptr i32 %t.1.in.i to ptr
  %.pn.in.in.i = ptrtoint ptr %.pn.in.in.in.i to i32
  %.pn3.in.i = xor i32 %.pn3.in.in.i, 32
  %.pn.in.i = xor i32 %.pn.in.in.i, 32
  %.pn3.i = inttoptr i32 %.pn3.in.i to ptr
  %.pn.i = inttoptr i32 %.pn.in.i to ptr
  %v1.1.in.i = getelementptr %struct.edge_rec, ptr %.pn3.i, i32 0, i32 0
  %v2.1.in.i = getelementptr %struct.edge_rec, ptr %.pn.i, i32 0, i32 0
  %v3.1.in.i = getelementptr %struct.edge_rec, ptr %rcand.2.pn.i, i32 0, i32 0
  %v1.1.i = load ptr, ptr %v1.1.in.i
  %v2.1.i = load ptr, ptr %v2.1.in.i
  %v3.1.i = load ptr, ptr %v3.1.in.i
  %452 = load double, ptr %363, align 4
  %453 = getelementptr %struct.VERTEX, ptr %v1.1.i, i32 0, i32 0, i32 0
  %454 = load double, ptr %453, align 4
  %455 = fsub double %454, %451
  %456 = getelementptr %struct.VERTEX, ptr %v1.1.i, i32 0, i32 0, i32 1
  %457 = load double, ptr %456, align 4
  %458 = fsub double %457, %450
  %459 = getelementptr %struct.VERTEX, ptr %v1.1.i, i32 0, i32 0, i32 2
  %460 = load double, ptr %459, align 4
  %461 = getelementptr %struct.VERTEX, ptr %v2.1.i, i32 0, i32 0, i32 0
  %462 = load double, ptr %461, align 4
  %463 = fsub double %462, %451
  %464 = getelementptr %struct.VERTEX, ptr %v2.1.i, i32 0, i32 0, i32 1
  %465 = load double, ptr %464, align 4
  %466 = fsub double %465, %450
  %467 = getelementptr %struct.VERTEX, ptr %v2.1.i, i32 0, i32 0, i32 2
  %468 = load double, ptr %467, align 4
  %469 = getelementptr %struct.VERTEX, ptr %v3.1.i, i32 0, i32 0, i32 0
  %470 = load double, ptr %469, align 4
  %471 = fsub double %470, %451
  %472 = getelementptr %struct.VERTEX, ptr %v3.1.i, i32 0, i32 0, i32 1
  %473 = load double, ptr %472, align 4
  %474 = fsub double %473, %450
  %475 = getelementptr %struct.VERTEX, ptr %v3.1.i, i32 0, i32 0, i32 2
  %476 = load double, ptr %475, align 4
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
  store ptr %avail_edge.tmp.0, ptr @avail_edge
  %.pre27.i = load ptr, ptr %170, align 4
  %.pre28.i = load ptr, ptr %175, align 4
  br label %bb17.i

bb17.i:
  %495 = phi ptr [ %.pre28.i, %bb17.loopexit.i ], [ %324, %bb13.i ]
  %496 = phi ptr [ %.pre27.i, %bb17.loopexit.i ], [ %325, %bb13.i ]
  %rcand.1.i = phi ptr [ %rcand.0.i, %bb17.loopexit.i ], [ %rcand.2.i, %bb13.i ]
  %497 = ptrtoint ptr %lcand.1.i to i32
  %498 = xor i32 %497, 32
  %499 = inttoptr i32 %498 to ptr
  %500 = getelementptr %struct.edge_rec, ptr %499, i32 0, i32 0
  %501 = load ptr, ptr %500, align 4
  %502 = getelementptr %struct.VERTEX, ptr %496, i32 0, i32 0, i32 0
  %503 = load double, ptr %502, align 4
  %504 = getelementptr %struct.VERTEX, ptr %496, i32 0, i32 0, i32 1
  %505 = load double, ptr %504, align 4
  %506 = getelementptr %struct.VERTEX, ptr %501, i32 0, i32 0, i32 0
  %507 = load double, ptr %506, align 4
  %508 = getelementptr %struct.VERTEX, ptr %501, i32 0, i32 0, i32 1
  %509 = load double, ptr %508, align 4
  %510 = getelementptr %struct.VERTEX, ptr %495, i32 0, i32 0, i32 0
  %511 = load double, ptr %510, align 4
  %512 = getelementptr %struct.VERTEX, ptr %495, i32 0, i32 0, i32 1
  %513 = load double, ptr %512, align 4
  %514 = fsub double %503, %511
  %515 = fsub double %509, %513
  %516 = fmul double %514, %515
  %517 = fsub double %507, %511
  %518 = fsub double %505, %513
  %519 = fmul double %517, %518
  %520 = fsub double %516, %519
  %521 = fcmp ogt double %520, 0.000000e+00
  %522 = ptrtoint ptr %rcand.1.i to i32
  %523 = xor i32 %522, 32
  %524 = inttoptr i32 %523 to ptr
  %525 = getelementptr %struct.edge_rec, ptr %524, i32 0, i32 0
  %526 = load ptr, ptr %525, align 4
  %527 = getelementptr %struct.VERTEX, ptr %526, i32 0, i32 0, i32 0
  %528 = load double, ptr %527, align 4
  %529 = getelementptr %struct.VERTEX, ptr %526, i32 0, i32 0, i32 1
  %530 = load double, ptr %529, align 4
  %531 = fsub double %530, %513
  %532 = fmul double %514, %531
  %533 = fsub double %528, %511
  %534 = fmul double %533, %518
  %535 = fsub double %532, %534
  %536 = fcmp ogt double %535, 0.000000e+00
  %537 = or i1 %536, %521
  br i1 %537, label %bb21.i, label %do_merge.exit

bb21.i:
  %538 = getelementptr %struct.edge_rec, ptr %lcand.1.i, i32 0, i32 0
  %539 = load ptr, ptr %538, align 4
  %540 = getelementptr %struct.edge_rec, ptr %rcand.1.i, i32 0, i32 0
  %541 = load ptr, ptr %540, align 4
  br i1 %521, label %bb22.i, label %bb24.i

bb22.i:
  br i1 %536, label %bb23.i, label %bb25.i

bb23.i:
  %542 = getelementptr %struct.VERTEX, ptr %526, i32 0, i32 0, i32 2
  %543 = load double, ptr %542, align 4
  %544 = fsub double %507, %528
  %545 = fsub double %509, %530
  %546 = getelementptr %struct.VERTEX, ptr %501, i32 0, i32 0, i32 2
  %547 = load double, ptr %546, align 4
  %548 = getelementptr %struct.VERTEX, ptr %539, i32 0, i32 0, i32 0
  %549 = load double, ptr %548, align 4
  %550 = fsub double %549, %528
  %551 = getelementptr %struct.VERTEX, ptr %539, i32 0, i32 0, i32 1
  %552 = load double, ptr %551, align 4
  %553 = fsub double %552, %530
  %554 = getelementptr %struct.VERTEX, ptr %539, i32 0, i32 0, i32 2
  %555 = load double, ptr %554, align 4
  %556 = getelementptr %struct.VERTEX, ptr %541, i32 0, i32 0, i32 0
  %557 = load double, ptr %556, align 4
  %558 = fsub double %557, %528
  %559 = getelementptr %struct.VERTEX, ptr %541, i32 0, i32 0, i32 1
  %560 = load double, ptr %559, align 4
  %561 = fsub double %560, %530
  %562 = getelementptr %struct.VERTEX, ptr %541, i32 0, i32 0, i32 2
  %563 = load double, ptr %562, align 4
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
  %586 = inttoptr i32 %585 to ptr
  %587 = getelementptr %struct.edge_rec, ptr %586, i32 0, i32 1
  %588 = load ptr, ptr %587, align 4
  %589 = ptrtoint ptr %588 to i32
  %590 = add i32 %589, 16
  %591 = and i32 %590, 63
  %592 = and i32 %589, -64
  %593 = or i32 %591, %592
  %594 = inttoptr i32 %593 to ptr
  %595 = call  ptr @alloc_edge() nounwind
  %596 = getelementptr %struct.edge_rec, ptr %595, i32 0, i32 1
  store ptr %595, ptr %596, align 4
  %597 = getelementptr %struct.edge_rec, ptr %595, i32 0, i32 0
  store ptr %526, ptr %597, align 4
  %598 = ptrtoint ptr %595 to i32
  %599 = add i32 %598, 16
  %600 = inttoptr i32 %599 to ptr
  %601 = add i32 %598, 48
  %602 = inttoptr i32 %601 to ptr
  %603 = getelementptr %struct.edge_rec, ptr %600, i32 0, i32 1
  store ptr %602, ptr %603, align 4
  %604 = add i32 %598, 32
  %605 = inttoptr i32 %604 to ptr
  %606 = getelementptr %struct.edge_rec, ptr %605, i32 0, i32 1
  store ptr %605, ptr %606, align 4
  %607 = getelementptr %struct.edge_rec, ptr %605, i32 0, i32 0
  store ptr %495, ptr %607, align 4
  %608 = getelementptr %struct.edge_rec, ptr %602, i32 0, i32 1
  store ptr %600, ptr %608, align 4
  %609 = load ptr, ptr %596, align 4
  %610 = ptrtoint ptr %609 to i32
  %611 = add i32 %610, 16
  %612 = and i32 %611, 63
  %613 = and i32 %610, -64
  %614 = or i32 %612, %613
  %615 = inttoptr i32 %614 to ptr
  %616 = getelementptr %struct.edge_rec, ptr %594, i32 0, i32 1
  %617 = load ptr, ptr %616, align 4
  %618 = ptrtoint ptr %617 to i32
  %619 = add i32 %618, 16
  %620 = and i32 %619, 63
  %621 = and i32 %618, -64
  %622 = or i32 %620, %621
  %623 = inttoptr i32 %622 to ptr
  %624 = getelementptr %struct.edge_rec, ptr %623, i32 0, i32 1
  %625 = load ptr, ptr %624, align 4
  %626 = getelementptr %struct.edge_rec, ptr %615, i32 0, i32 1
  %627 = load ptr, ptr %626, align 4
  store ptr %625, ptr %626, align 4
  store ptr %627, ptr %624, align 4
  %628 = load ptr, ptr %596, align 4
  %629 = load ptr, ptr %616, align 4
  store ptr %628, ptr %616, align 4
  store ptr %629, ptr %596, align 4
  %630 = xor i32 %598, 32
  %631 = inttoptr i32 %630 to ptr
  %632 = getelementptr %struct.edge_rec, ptr %631, i32 0, i32 1
  %633 = load ptr, ptr %632, align 4
  %634 = ptrtoint ptr %633 to i32
  %635 = add i32 %634, 16
  %636 = and i32 %635, 63
  %637 = and i32 %634, -64
  %638 = or i32 %636, %637
  %639 = inttoptr i32 %638 to ptr
  %640 = getelementptr %struct.edge_rec, ptr %174, i32 0, i32 1
  %641 = load ptr, ptr %640, align 4
  %642 = ptrtoint ptr %641 to i32
  %643 = add i32 %642, 16
  %644 = and i32 %643, 63
  %645 = and i32 %642, -64
  %646 = or i32 %644, %645
  %647 = inttoptr i32 %646 to ptr
  %648 = getelementptr %struct.edge_rec, ptr %647, i32 0, i32 1
  %649 = load ptr, ptr %648, align 4
  %650 = getelementptr %struct.edge_rec, ptr %639, i32 0, i32 1
  %651 = load ptr, ptr %650, align 4
  store ptr %649, ptr %650, align 4
  store ptr %651, ptr %648, align 4
  %652 = load ptr, ptr %632, align 4
  %653 = load ptr, ptr %640, align 4
  store ptr %652, ptr %640, align 4
  store ptr %653, ptr %632, align 4
  %654 = add i32 %630, 48
  %655 = and i32 %654, 63
  %656 = and i32 %598, -64
  %657 = or i32 %655, %656
  %658 = inttoptr i32 %657 to ptr
  %659 = getelementptr %struct.edge_rec, ptr %658, i32 0, i32 1
  %660 = load ptr, ptr %659, align 4
  %661 = ptrtoint ptr %660 to i32
  %662 = add i32 %661, 16
  %663 = and i32 %662, 63
  %664 = and i32 %661, -64
  %665 = or i32 %663, %664
  %666 = inttoptr i32 %665 to ptr
  br label %bb9.i

bb25.i:
  %667 = add i32 %172, 16
  %668 = and i32 %667, 63
  %669 = and i32 %172, -64
  %670 = or i32 %668, %669
  %671 = inttoptr i32 %670 to ptr
  %672 = getelementptr %struct.edge_rec, ptr %671, i32 0, i32 1
  %673 = load ptr, ptr %672, align 4
  %674 = ptrtoint ptr %673 to i32
  %675 = add i32 %674, 16
  %676 = and i32 %675, 63
  %677 = and i32 %674, -64
  %678 = or i32 %676, %677
  %679 = inttoptr i32 %678 to ptr
  %680 = call  ptr @alloc_edge() nounwind
  %681 = getelementptr %struct.edge_rec, ptr %680, i32 0, i32 1
  store ptr %680, ptr %681, align 4
  %682 = getelementptr %struct.edge_rec, ptr %680, i32 0, i32 0
  store ptr %501, ptr %682, align 4
  %683 = ptrtoint ptr %680 to i32
  %684 = add i32 %683, 16
  %685 = inttoptr i32 %684 to ptr
  %686 = add i32 %683, 48
  %687 = inttoptr i32 %686 to ptr
  %688 = getelementptr %struct.edge_rec, ptr %685, i32 0, i32 1
  store ptr %687, ptr %688, align 4
  %689 = add i32 %683, 32
  %690 = inttoptr i32 %689 to ptr
  %691 = getelementptr %struct.edge_rec, ptr %690, i32 0, i32 1
  store ptr %690, ptr %691, align 4
  %692 = getelementptr %struct.edge_rec, ptr %690, i32 0, i32 0
  store ptr %496, ptr %692, align 4
  %693 = getelementptr %struct.edge_rec, ptr %687, i32 0, i32 1
  store ptr %685, ptr %693, align 4
  %694 = load ptr, ptr %681, align 4
  %695 = ptrtoint ptr %694 to i32
  %696 = add i32 %695, 16
  %697 = and i32 %696, 63
  %698 = and i32 %695, -64
  %699 = or i32 %697, %698
  %700 = inttoptr i32 %699 to ptr
  %701 = getelementptr %struct.edge_rec, ptr %499, i32 0, i32 1
  %702 = load ptr, ptr %701, align 4
  %703 = ptrtoint ptr %702 to i32
  %704 = add i32 %703, 16
  %705 = and i32 %704, 63
  %706 = and i32 %703, -64
  %707 = or i32 %705, %706
  %708 = inttoptr i32 %707 to ptr
  %709 = getelementptr %struct.edge_rec, ptr %708, i32 0, i32 1
  %710 = load ptr, ptr %709, align 4
  %711 = getelementptr %struct.edge_rec, ptr %700, i32 0, i32 1
  %712 = load ptr, ptr %711, align 4
  store ptr %710, ptr %711, align 4
  store ptr %712, ptr %709, align 4
  %713 = load ptr, ptr %681, align 4
  %714 = load ptr, ptr %701, align 4
  store ptr %713, ptr %701, align 4
  store ptr %714, ptr %681, align 4
  %715 = xor i32 %683, 32
  %716 = inttoptr i32 %715 to ptr
  %717 = getelementptr %struct.edge_rec, ptr %716, i32 0, i32 1
  %718 = load ptr, ptr %717, align 4
  %719 = ptrtoint ptr %718 to i32
  %720 = add i32 %719, 16
  %721 = and i32 %720, 63
  %722 = and i32 %719, -64
  %723 = or i32 %721, %722
  %724 = inttoptr i32 %723 to ptr
  %725 = getelementptr %struct.edge_rec, ptr %679, i32 0, i32 1
  %726 = load ptr, ptr %725, align 4
  %727 = ptrtoint ptr %726 to i32
  %728 = add i32 %727, 16
  %729 = and i32 %728, 63
  %730 = and i32 %727, -64
  %731 = or i32 %729, %730
  %732 = inttoptr i32 %731 to ptr
  %733 = getelementptr %struct.edge_rec, ptr %732, i32 0, i32 1
  %734 = load ptr, ptr %733, align 4
  %735 = getelementptr %struct.edge_rec, ptr %724, i32 0, i32 1
  %736 = load ptr, ptr %735, align 4
  store ptr %734, ptr %735, align 4
  store ptr %736, ptr %733, align 4
  %737 = load ptr, ptr %717, align 4
  %738 = load ptr, ptr %725, align 4
  store ptr %737, ptr %725, align 4
  store ptr %738, ptr %717, align 4
  %739 = load ptr, ptr %681, align 4
  br label %bb9.i

do_merge.exit:
  %740 = getelementptr %struct.edge_rec, ptr %ldo_addr.0.ph.i, i32 0, i32 0
  %741 = load ptr, ptr %740, align 4
  %742 = icmp eq ptr %741, %tree_addr.0.i
  br i1 %742, label %bb5.loopexit, label %bb2

bb2:
  %ldo.07 = phi ptr [ %747, %bb2 ], [ %ldo_addr.0.ph.i, %do_merge.exit ]
  %743 = ptrtoint ptr %ldo.07 to i32
  %744 = xor i32 %743, 32
  %745 = inttoptr i32 %744 to ptr
  %746 = getelementptr %struct.edge_rec, ptr %745, i32 0, i32 1
  %747 = load ptr, ptr %746, align 4
  %748 = getelementptr %struct.edge_rec, ptr %747, i32 0, i32 0
  %749 = load ptr, ptr %748, align 4
  %750 = icmp eq ptr %749, %tree_addr.0.i
  br i1 %750, label %bb5.loopexit, label %bb2

bb4:
  %rdo.05 = phi ptr [ %755, %bb4 ], [ %rdo_addr.0.i, %bb5.loopexit ]
  %751 = getelementptr %struct.edge_rec, ptr %rdo.05, i32 0, i32 1
  %752 = load ptr, ptr %751, align 4
  %753 = ptrtoint ptr %752 to i32
  %754 = xor i32 %753, 32
  %755 = inttoptr i32 %754 to ptr
  %756 = getelementptr %struct.edge_rec, ptr %755, i32 0, i32 0
  %757 = load ptr, ptr %756, align 4
  %758 = icmp eq ptr %757, %extra
  br i1 %758, label %bb6, label %bb4

bb5.loopexit:
  %ldo.0.lcssa = phi ptr [ %ldo_addr.0.ph.i, %do_merge.exit ], [ %747, %bb2 ]
  %759 = getelementptr %struct.edge_rec, ptr %rdo_addr.0.i, i32 0, i32 0
  %760 = load ptr, ptr %759, align 4
  %761 = icmp eq ptr %760, %extra
  br i1 %761, label %bb6, label %bb4

bb6:
  %rdo.0.lcssa = phi ptr [ %rdo_addr.0.i, %bb5.loopexit ], [ %755, %bb4 ]
  %tmp16 = ptrtoint ptr %ldo.0.lcssa to i32
  %tmp4 = ptrtoint ptr %rdo.0.lcssa to i32
  br label %bb15

bb7:
  %762 = getelementptr %struct.VERTEX, ptr %tree, i32 0, i32 1
  %763 = load ptr, ptr %762, align 4
  %764 = icmp eq ptr %763, null
  %765 = call  ptr @alloc_edge() nounwind
  %766 = getelementptr %struct.edge_rec, ptr %765, i32 0, i32 1
  store ptr %765, ptr %766, align 4
  %767 = getelementptr %struct.edge_rec, ptr %765, i32 0, i32 0
  br i1 %764, label %bb10, label %bb11

bb8:
  %768 = call  i32 @puts(ptr @_2E_str7) nounwind
  call  void @exit(i32 -1) noreturn nounwind
  unreachable

bb10:
  store ptr %tree, ptr %767, align 4
  %769 = ptrtoint ptr %765 to i32
  %770 = add i32 %769, 16
  %771 = inttoptr i32 %770 to ptr
  %772 = add i32 %769, 48
  %773 = inttoptr i32 %772 to ptr
  %774 = getelementptr %struct.edge_rec, ptr %771, i32 0, i32 1
  store ptr %773, ptr %774, align 4
  %775 = add i32 %769, 32
  %776 = inttoptr i32 %775 to ptr
  %777 = getelementptr %struct.edge_rec, ptr %776, i32 0, i32 1
  store ptr %776, ptr %777, align 4
  %778 = getelementptr %struct.edge_rec, ptr %776, i32 0, i32 0
  store ptr %extra, ptr %778, align 4
  %779 = getelementptr %struct.edge_rec, ptr %773, i32 0, i32 1
  store ptr %771, ptr %779, align 4
  %780 = xor i32 %769, 32
  br label %bb15

bb11:
  store ptr %763, ptr %767, align 4
  %781 = ptrtoint ptr %765 to i32
  %782 = add i32 %781, 16
  %783 = inttoptr i32 %782 to ptr
  %784 = add i32 %781, 48
  %785 = inttoptr i32 %784 to ptr
  %786 = getelementptr %struct.edge_rec, ptr %783, i32 0, i32 1
  store ptr %785, ptr %786, align 4
  %787 = add i32 %781, 32
  %788 = inttoptr i32 %787 to ptr
  %789 = getelementptr %struct.edge_rec, ptr %788, i32 0, i32 1
  store ptr %788, ptr %789, align 4
  %790 = getelementptr %struct.edge_rec, ptr %788, i32 0, i32 0
  store ptr %tree, ptr %790, align 4
  %791 = getelementptr %struct.edge_rec, ptr %785, i32 0, i32 1
  store ptr %783, ptr %791, align 4
  %792 = call  ptr @alloc_edge() nounwind
  %793 = getelementptr %struct.edge_rec, ptr %792, i32 0, i32 1
  store ptr %792, ptr %793, align 4
  %794 = getelementptr %struct.edge_rec, ptr %792, i32 0, i32 0
  store ptr %tree, ptr %794, align 4
  %795 = ptrtoint ptr %792 to i32
  %796 = add i32 %795, 16
  %797 = inttoptr i32 %796 to ptr
  %798 = add i32 %795, 48
  %799 = inttoptr i32 %798 to ptr
  %800 = getelementptr %struct.edge_rec, ptr %797, i32 0, i32 1
  store ptr %799, ptr %800, align 4
  %801 = add i32 %795, 32
  %802 = inttoptr i32 %801 to ptr
  %803 = getelementptr %struct.edge_rec, ptr %802, i32 0, i32 1
  store ptr %802, ptr %803, align 4
  %804 = getelementptr %struct.edge_rec, ptr %802, i32 0, i32 0
  store ptr %extra, ptr %804, align 4
  %805 = getelementptr %struct.edge_rec, ptr %799, i32 0, i32 1
  store ptr %797, ptr %805, align 4
  %806 = xor i32 %781, 32
  %807 = inttoptr i32 %806 to ptr
  %808 = getelementptr %struct.edge_rec, ptr %807, i32 0, i32 1
  %809 = load ptr, ptr %808, align 4
  %810 = ptrtoint ptr %809 to i32
  %811 = add i32 %810, 16
  %812 = and i32 %811, 63
  %813 = and i32 %810, -64
  %814 = or i32 %812, %813
  %815 = inttoptr i32 %814 to ptr
  %816 = load ptr, ptr %793, align 4
  %817 = ptrtoint ptr %816 to i32
  %818 = add i32 %817, 16
  %819 = and i32 %818, 63
  %820 = and i32 %817, -64
  %821 = or i32 %819, %820
  %822 = inttoptr i32 %821 to ptr
  %823 = getelementptr %struct.edge_rec, ptr %822, i32 0, i32 1
  %824 = load ptr, ptr %823, align 4
  %825 = getelementptr %struct.edge_rec, ptr %815, i32 0, i32 1
  %826 = load ptr, ptr %825, align 4
  store ptr %824, ptr %825, align 4
  store ptr %826, ptr %823, align 4
  %827 = load ptr, ptr %808, align 4
  %828 = load ptr, ptr %793, align 4
  store ptr %827, ptr %793, align 4
  store ptr %828, ptr %808, align 4
  %829 = xor i32 %795, 32
  %830 = inttoptr i32 %829 to ptr
  %831 = getelementptr %struct.edge_rec, ptr %830, i32 0, i32 0
  %832 = load ptr, ptr %831, align 4
  %833 = and i32 %798, 63
  %834 = and i32 %795, -64
  %835 = or i32 %833, %834
  %836 = inttoptr i32 %835 to ptr
  %837 = getelementptr %struct.edge_rec, ptr %836, i32 0, i32 1
  %838 = load ptr, ptr %837, align 4
  %839 = ptrtoint ptr %838 to i32
  %840 = add i32 %839, 16
  %841 = and i32 %840, 63
  %842 = and i32 %839, -64
  %843 = or i32 %841, %842
  %844 = inttoptr i32 %843 to ptr
  %845 = load ptr, ptr %767, align 4
  %846 = call  ptr @alloc_edge() nounwind
  %847 = getelementptr %struct.edge_rec, ptr %846, i32 0, i32 1
  store ptr %846, ptr %847, align 4
  %848 = getelementptr %struct.edge_rec, ptr %846, i32 0, i32 0
  store ptr %832, ptr %848, align 4
  %849 = ptrtoint ptr %846 to i32
  %850 = add i32 %849, 16
  %851 = inttoptr i32 %850 to ptr
  %852 = add i32 %849, 48
  %853 = inttoptr i32 %852 to ptr
  %854 = getelementptr %struct.edge_rec, ptr %851, i32 0, i32 1
  store ptr %853, ptr %854, align 4
  %855 = add i32 %849, 32
  %856 = inttoptr i32 %855 to ptr
  %857 = getelementptr %struct.edge_rec, ptr %856, i32 0, i32 1
  store ptr %856, ptr %857, align 4
  %858 = getelementptr %struct.edge_rec, ptr %856, i32 0, i32 0
  store ptr %845, ptr %858, align 4
  %859 = getelementptr %struct.edge_rec, ptr %853, i32 0, i32 1
  store ptr %851, ptr %859, align 4
  %860 = load ptr, ptr %847, align 4
  %861 = ptrtoint ptr %860 to i32
  %862 = add i32 %861, 16
  %863 = and i32 %862, 63
  %864 = and i32 %861, -64
  %865 = or i32 %863, %864
  %866 = inttoptr i32 %865 to ptr
  %867 = getelementptr %struct.edge_rec, ptr %844, i32 0, i32 1
  %868 = load ptr, ptr %867, align 4
  %869 = ptrtoint ptr %868 to i32
  %870 = add i32 %869, 16
  %871 = and i32 %870, 63
  %872 = and i32 %869, -64
  %873 = or i32 %871, %872
  %874 = inttoptr i32 %873 to ptr
  %875 = getelementptr %struct.edge_rec, ptr %874, i32 0, i32 1
  %876 = load ptr, ptr %875, align 4
  %877 = getelementptr %struct.edge_rec, ptr %866, i32 0, i32 1
  %878 = load ptr, ptr %877, align 4
  store ptr %876, ptr %877, align 4
  store ptr %878, ptr %875, align 4
  %879 = load ptr, ptr %847, align 4
  %880 = load ptr, ptr %867, align 4
  store ptr %879, ptr %867, align 4
  store ptr %880, ptr %847, align 4
  %881 = xor i32 %849, 32
  %882 = inttoptr i32 %881 to ptr
  %883 = getelementptr %struct.edge_rec, ptr %882, i32 0, i32 1
  %884 = load ptr, ptr %883, align 4
  %885 = ptrtoint ptr %884 to i32
  %886 = add i32 %885, 16
  %887 = and i32 %886, 63
  %888 = and i32 %885, -64
  %889 = or i32 %887, %888
  %890 = inttoptr i32 %889 to ptr
  %891 = load ptr, ptr %766, align 4
  %892 = ptrtoint ptr %891 to i32
  %893 = add i32 %892, 16
  %894 = and i32 %893, 63
  %895 = and i32 %892, -64
  %896 = or i32 %894, %895
  %897 = inttoptr i32 %896 to ptr
  %898 = getelementptr %struct.edge_rec, ptr %897, i32 0, i32 1
  %899 = load ptr, ptr %898, align 4
  %900 = getelementptr %struct.edge_rec, ptr %890, i32 0, i32 1
  %901 = load ptr, ptr %900, align 4
  store ptr %899, ptr %900, align 4
  store ptr %901, ptr %898, align 4
  %902 = load ptr, ptr %883, align 4
  %903 = load ptr, ptr %766, align 4
  store ptr %902, ptr %766, align 4
  store ptr %903, ptr %883, align 4
  %904 = getelementptr %struct.VERTEX, ptr %763, i32 0, i32 0, i32 0
  %905 = load double, ptr %904, align 4
  %906 = getelementptr %struct.VERTEX, ptr %763, i32 0, i32 0, i32 1
  %907 = load double, ptr %906, align 4
  %908 = getelementptr %struct.VERTEX, ptr %extra, i32 0, i32 0, i32 0
  %909 = load double, ptr %908, align 4
  %910 = getelementptr %struct.VERTEX, ptr %extra, i32 0, i32 0, i32 1
  %911 = load double, ptr %910, align 4
  %912 = getelementptr %struct.VERTEX, ptr %tree, i32 0, i32 0, i32 0
  %913 = load double, ptr %912, align 4
  %914 = getelementptr %struct.VERTEX, ptr %tree, i32 0, i32 0, i32 1
  %915 = load double, ptr %914, align 4
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
  %935 = inttoptr i32 %934 to ptr
  %936 = getelementptr %struct.edge_rec, ptr %935, i32 0, i32 1
  %937 = load ptr, ptr %936, align 4
  %938 = ptrtoint ptr %937 to i32
  %939 = add i32 %938, 16
  %940 = and i32 %939, 63
  %941 = and i32 %938, -64
  %942 = or i32 %940, %941
  %943 = inttoptr i32 %942 to ptr
  %944 = load ptr, ptr %847, align 4
  %945 = ptrtoint ptr %944 to i32
  %946 = add i32 %945, 16
  %947 = and i32 %946, 63
  %948 = and i32 %945, -64
  %949 = or i32 %947, %948
  %950 = inttoptr i32 %949 to ptr
  %951 = getelementptr %struct.edge_rec, ptr %943, i32 0, i32 1
  %952 = load ptr, ptr %951, align 4
  %953 = ptrtoint ptr %952 to i32
  %954 = add i32 %953, 16
  %955 = and i32 %954, 63
  %956 = and i32 %953, -64
  %957 = or i32 %955, %956
  %958 = inttoptr i32 %957 to ptr
  %959 = getelementptr %struct.edge_rec, ptr %958, i32 0, i32 1
  %960 = load ptr, ptr %959, align 4
  %961 = getelementptr %struct.edge_rec, ptr %950, i32 0, i32 1
  %962 = load ptr, ptr %961, align 4
  store ptr %960, ptr %961, align 4
  store ptr %962, ptr %959, align 4
  %963 = load ptr, ptr %847, align 4
  %964 = load ptr, ptr %951, align 4
  store ptr %963, ptr %951, align 4
  store ptr %964, ptr %847, align 4
  %965 = add i32 %881, 16
  %966 = and i32 %965, 63
  %967 = or i32 %966, %933
  %968 = inttoptr i32 %967 to ptr
  %969 = getelementptr %struct.edge_rec, ptr %968, i32 0, i32 1
  %970 = load ptr, ptr %969, align 4
  %971 = ptrtoint ptr %970 to i32
  %972 = add i32 %971, 16
  %973 = and i32 %972, 63
  %974 = and i32 %971, -64
  %975 = or i32 %973, %974
  %976 = inttoptr i32 %975 to ptr
  %977 = load ptr, ptr %883, align 4
  %978 = ptrtoint ptr %977 to i32
  %979 = add i32 %978, 16
  %980 = and i32 %979, 63
  %981 = and i32 %978, -64
  %982 = or i32 %980, %981
  %983 = inttoptr i32 %982 to ptr
  %984 = getelementptr %struct.edge_rec, ptr %976, i32 0, i32 1
  %985 = load ptr, ptr %984, align 4
  %986 = ptrtoint ptr %985 to i32
  %987 = add i32 %986, 16
  %988 = and i32 %987, 63
  %989 = and i32 %986, -64
  %990 = or i32 %988, %989
  %991 = inttoptr i32 %990 to ptr
  %992 = getelementptr %struct.edge_rec, ptr %991, i32 0, i32 1
  %993 = load ptr, ptr %992, align 4
  %994 = getelementptr %struct.edge_rec, ptr %983, i32 0, i32 1
  %995 = load ptr, ptr %994, align 4
  store ptr %993, ptr %994, align 4
  store ptr %995, ptr %992, align 4
  %996 = load ptr, ptr %883, align 4
  %997 = load ptr, ptr %984, align 4
  store ptr %996, ptr %984, align 4
  store ptr %997, ptr %883, align 4
  %998 = inttoptr i32 %933 to ptr
  %999 = load ptr, ptr @avail_edge, align 4
  %1000 = getelementptr %struct.edge_rec, ptr %998, i32 0, i32 1
  store ptr %999, ptr %1000, align 4
  store ptr %998, ptr @avail_edge, align 4
  br label %bb15

bb15:
  %retval.1.0 = phi i32 [ %780, %bb10 ], [ %829, %bb13 ], [ %829, %bb14 ], [ %tmp4, %bb6 ], [ %849, %bb11 ]
  %retval.0.0 = phi i32 [ %769, %bb10 ], [ %781, %bb13 ], [ %781, %bb14 ], [ %tmp16, %bb6 ], [ %881, %bb11 ]
  %agg.result162 = bitcast ptr %agg.result to ptr
  %1001 = zext i32 %retval.0.0 to i64
  %1002 = zext i32 %retval.1.0 to i64
  %1003 = shl i64 %1002, 32
  %1004 = or i64 %1003, %1001
  store i64 %1004, ptr %agg.result162, align 4
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

declare i32 @puts(ptr nocapture) nounwind

declare void @exit(i32) noreturn nounwind

declare ptr @alloc_edge() nounwind
