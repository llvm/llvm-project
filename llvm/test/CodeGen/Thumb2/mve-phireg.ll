; RUN: llc -O3 -mtriple=thumbv8.1m.main-arm-none-eabi -mattr=+mve.fp -verify-machineinstrs %s -o - | FileCheck %s

; verify-machineinstrs previously caught the incorrect use of QPR in the stack reloads.

define arm_aapcs_vfpcc void @k() {
; CHECK-LABEL: k:
; CHECK:    vstrw.32
; CHECK:    vldrw.u32
entry:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %entry
  %vec.ind = phi <8 x i32> [ <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, %entry ], [ zeroinitializer, %vector.body ]
  %0 = and <8 x i32> %vec.ind, <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  %1 = icmp eq <8 x i32> %0, zeroinitializer
  %2 = select <8 x i1> %1, <8 x i16> <i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3>, <8 x i16> <i16 6, i16 6, i16 6, i16 6, i16 6, i16 6, i16 6, i16 6>
  %3 = bitcast i16* undef to <8 x i16>*
  store <8 x i16> %2, <8 x i16>* %3, align 2
  %4 = icmp eq i32 undef, 128
  br i1 %4, label %for.cond4.preheader, label %vector.body

for.cond4.preheader:                              ; preds = %vector.body
  br i1 undef, label %vector.body105, label %for.body10

for.cond4.loopexit:                               ; preds = %for.body10
  %call5 = call arm_aapcs_vfpcc i32 bitcast (i32 (...)* @l to i32 ()*)()
  br label %vector.body105

for.body10:                                       ; preds = %for.body10, %for.cond4.preheader
  %exitcond88 = icmp eq i32 undef, 7
  br i1 %exitcond88, label %for.cond4.loopexit, label %for.body10

vector.body105:                                   ; preds = %vector.body105, %for.cond4.loopexit, %for.cond4.preheader
  %vec.ind113 = phi <8 x i32> [ %vec.ind.next114, %vector.body105 ], [ <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, %for.cond4.loopexit ], [ <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, %for.cond4.preheader ]
  %5 = and <8 x i32> %vec.ind113, <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  %vec.ind.next114 = add <8 x i32> %vec.ind113, <i32 8, i32 8, i32 8, i32 8, i32 8, i32 8, i32 8, i32 8>
  %6 = icmp eq i32 undef, 256
  br i1 %6, label %vector.body115.ph, label %vector.body105

vector.body115.ph:                                ; preds = %vector.body105
  tail call void asm sideeffect "nop", "~{s0},~{s4},~{s8},~{s12},~{s16},~{s20},~{s24},~{s28},~{memory}"()
  br label %vector.body115

vector.body115:                                   ; preds = %vector.body115, %vector.body115.ph
  %vec.ind123 = phi <4 x i32> [ %vec.ind.next124, %vector.body115 ], [ <i32 0, i32 1, i32 2, i32 3>, %vector.body115.ph ]
  %7 = icmp eq <4 x i32> %vec.ind123, zeroinitializer
  %vec.ind.next124 = add <4 x i32> %vec.ind123, <i32 4, i32 4, i32 4, i32 4>
  br label %vector.body115
}


@a = external dso_local global i32, align 4
@b = dso_local local_unnamed_addr global i32 ptrtoint (i32* @a to i32), align 4
@c = dso_local global i32 2, align 4
@d = dso_local global i32 2, align 4

define dso_local i32 @e() #0 {
; CHECK-LABEL: e:
; CHECK:    vstrw.32
; CHECK:    vldrw.u32
entry:
  %f = alloca i16, align 2
  %g = alloca [3 x [8 x [4 x i16*]]], align 4
  store i16 4, i16* %f, align 2
  %0 = load i32, i32* @c, align 4
  %1 = load i32, i32* @d, align 4
  %arrayinit.element7 = getelementptr inbounds [3 x [8 x [4 x i16*]]], [3 x [8 x [4 x i16*]]]* %g, i32 0, i32 0, i32 1, i32 1
  %2 = bitcast i16** %arrayinit.element7 to i32*
  store i32 %0, i32* %2, align 4
  %arrayinit.element8 = getelementptr inbounds [3 x [8 x [4 x i16*]]], [3 x [8 x [4 x i16*]]]* %g, i32 0, i32 0, i32 1, i32 2
  store i16* null, i16** %arrayinit.element8, align 4
  %3 = bitcast i16** undef to i32*
  store i32 %1, i32* %3, align 4
  %4 = bitcast i16** undef to i32*
  store i32 %0, i32* %4, align 4
  %arrayinit.element13 = getelementptr inbounds [3 x [8 x [4 x i16*]]], [3 x [8 x [4 x i16*]]]* %g, i32 0, i32 0, i32 2, i32 2
  %5 = bitcast i16** %arrayinit.element13 to <4 x i16*>*
  store <4 x i16*> <i16* inttoptr (i32 4 to i16*), i16* bitcast (i32* @c to i16*), i16* bitcast (i32* @c to i16*), i16* null>, <4 x i16*>* %5, align 4
  %arrayinit.element24 = getelementptr inbounds [3 x [8 x [4 x i16*]]], [3 x [8 x [4 x i16*]]]* %g, i32 0, i32 0, i32 4, i32 2
  %6 = bitcast i16** %arrayinit.element24 to <4 x i16*>*
  store <4 x i16*> <i16* bitcast (i32* @d to i16*), i16* null, i16* bitcast (i32* @d to i16*), i16* bitcast (i32 ()* @e to i16*)>, <4 x i16*>* %6, align 4
  %7 = bitcast i16** undef to <4 x i16*>*
  store <4 x i16*> <i16* inttoptr (i32 4 to i16*), i16* bitcast (i32 ()* @e to i16*), i16* bitcast (i32* @c to i16*), i16* null>, <4 x i16*>* %7, align 4
  %8 = bitcast i16** undef to <4 x i16*>*
  store <4 x i16*> <i16* bitcast (i32* @c to i16*), i16* bitcast (i32 ()* @e to i16*), i16* bitcast (i32* @c to i16*), i16* bitcast (i32* @c to i16*)>, <4 x i16*>* %8, align 4
  %9 = bitcast i16** undef to <4 x i16*>*
  store <4 x i16*> <i16* bitcast (i32 ()* @e to i16*), i16* bitcast (i32* @c to i16*), i16* bitcast (i32* @c to i16*), i16* bitcast (i32* @c to i16*)>, <4 x i16*>* %9, align 4
  %10 = bitcast i16** undef to <4 x i16*>*
  store <4 x i16*> <i16* bitcast (i32* @c to i16*), i16* bitcast (i32* @c to i16*), i16* null, i16* bitcast (i32 ()* @e to i16*)>, <4 x i16*>* %10, align 4
  call void @llvm.memset.p0i8.i32(i8* nonnull align 4 dereferenceable(64) undef, i8 0, i32 64, i1 false)
  %11 = bitcast i16** undef to <4 x i16*>*
  store <4 x i16*> <i16* bitcast (i32* @d to i16*), i16* bitcast (i32 ()* @e to i16*), i16* bitcast (i32* @c to i16*), i16* bitcast (i32* @d to i16*)>, <4 x i16*>* %11, align 4
  %12 = bitcast i16** undef to <4 x i16*>*
  store <4 x i16*> <i16* null, i16* bitcast (i32* @c to i16*), i16* bitcast (i32* @c to i16*), i16* bitcast (i32* @c to i16*)>, <4 x i16*>* %12, align 4
  %13 = bitcast i16** undef to <4 x i16*>*
  store <4 x i16*> <i16* bitcast (i32* @c to i16*), i16* bitcast (i32* @d to i16*), i16* bitcast (i32* @c to i16*), i16* null>, <4 x i16*>* %13, align 4
  %arrayinit.begin78 = getelementptr inbounds [3 x [8 x [4 x i16*]]], [3 x [8 x [4 x i16*]]]* %g, i32 0, i32 2, i32 3, i32 0
  store i16* inttoptr (i32 4 to i16*), i16** %arrayinit.begin78, align 4
  store i32 0, i32* @b, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.cond, %entry
  br label %for.cond
}

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.memset.p0i8.i32(i8* nocapture writeonly, i8, i32, i1 immarg) #1

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.memset.p0i8.i64(i8* nocapture writeonly, i8, i64, i1 immarg) #1


declare arm_aapcs_vfpcc i32 @l(...)
