; RUN: llc -march=mipsel -O3 -relocation-model=pic < %s | FileCheck %s


; MIPS direct branches implicitly define register $at. This test makes sure that
; code hoisting optimization (which moves identical instructions at the start of
; two basic blocks to the common predecessor block) takes this into account and
; doesn't move definition of $at to the predecessor block (which would make $at
; live-in at the start of successor block).


; CHECK-LABEL: readLumaCoeff8x8_CABAC

; The check for first "addiu" instruction is added so that we can match the correct "b" instruction.
; CHECK:           andi
; CHECK:           b $[[BB0:BB[0-9_]+]]
; CHECK:           sll

; Check that at the start of a fallthrough block there is a instruction that writes to $1.
; CHECK:       {{BB[0-9_#]+}}:
; CHECK:           sll $1, $[[R0:[0-9]+]], 4
; CHECK:           lw      $[[R1:[0-9]+]], %got(assignSE2partition)($[[R2:[0-9]+]])

; Check that identical instructions are at the start of a target block.
; CHECK:       [[BB0]]:
; CHECK:           sll $1, $[[R0]], 4
; CHECK:           lw      $[[R1]], %got(assignSE2partition)($[[R2]])


%struct.img_par = type { i32, i32, i32, i32, i32*, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, [16 x [16 x i16]], [6 x [32 x i32]], [16 x [16 x i32]], [4 x [12 x [4 x [4 x i32]]]], [16 x i32], i8**, i32*, i32***, i32**, i32, i32, i32, i32, %struct.Slice*, %struct.macroblock*, i32, i32, i32, i32, i32, i32, %struct.DecRefPicMarking_s*, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, [3 x i32], i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32***, i32***, i32****, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, [3 x [2 x i32]], [3 x [2 x i32]], i32, i32, i32, i32, %struct.timeb, %struct.timeb, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 }
%struct.Slice = type { i32, i32, i32, i32, i32, i32, i32, i32, i32, %struct.datapartition*, %struct.MotionInfoContexts*, %struct.TextureInfoContexts*, i32, i32*, i32*, i32*, i32, i32*, i32*, i32*, i32 (%struct.img_par*, %struct.inp_par*)*, i32, i32, i32, i32 }
%struct.datapartition = type { %struct.Bitstream*, %struct.DecodingEnvironment, i32 (%struct.syntaxelement*, %struct.img_par*, %struct.datapartition*)* }
%struct.Bitstream = type { i32, i32, i32, i32, i8*, i32 }
%struct.DecodingEnvironment = type { i32, i32, i32, i32, i32, i8*, i32* }
%struct.syntaxelement = type { i32, i32, i32, i32, i32, i32, i32, i32, void (i32, i32, i32*, i32*)*, void (%struct.syntaxelement*, %struct.img_par*, %struct.DecodingEnvironment*)* }
%struct.MotionInfoContexts = type { [4 x [11 x %struct.BiContextType]], [2 x [9 x %struct.BiContextType]], [2 x [10 x %struct.BiContextType]], [2 x [6 x %struct.BiContextType]], [4 x %struct.BiContextType], [4 x %struct.BiContextType], [3 x %struct.BiContextType] }
%struct.BiContextType = type { i16, i8 }
%struct.TextureInfoContexts = type { [2 x %struct.BiContextType], [4 x %struct.BiContextType], [3 x [4 x %struct.BiContextType]], [10 x [4 x %struct.BiContextType]], [10 x [15 x %struct.BiContextType]], [10 x [15 x %struct.BiContextType]], [10 x [5 x %struct.BiContextType]], [10 x [5 x %struct.BiContextType]], [10 x [15 x %struct.BiContextType]], [10 x [15 x %struct.BiContextType]] }
%struct.inp_par = type { [1000 x i8], [1000 x i8], [1000 x i8], i32, i32, i32, i32, i32, i32, i32, i32 }
%struct.macroblock = type { i32, [2 x i32], i32, i32, %struct.macroblock*, %struct.macroblock*, i32, [2 x [4 x [4 x [2 x i32]]]], i32, i64, i64, i32, i32, [4 x i8], [4 x i8], i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 }
%struct.DecRefPicMarking_s = type { i32, i32, i32, i32, i32, %struct.DecRefPicMarking_s* }
%struct.timeb = type { i32, i16, i16, i16 }

@assignSE2partition = external global [0 x [20 x i32]]
@FIELD_SCAN8x8 = external constant [64 x [2 x i8]]


define void @readLumaCoeff8x8_CABAC(%struct.img_par* %img, i32 %b8) {

  %1 = load i32, i32* undef, align 4
  br i1 false, label %2, label %3

; <label>:2                                       ; preds = %0
  br label %3

; <label>:3                                       ; preds = %2, %0
  br i1 undef, label %switch.lookup, label %4

switch.lookup:                                    ; preds = %3
  br label %4

; <label>:4                                       ; preds = %switch.lookup, %3
  br i1 undef, label %5, label %6

; <label>:5                                       ; preds = %4
  br label %6

; <label>:6                                       ; preds = %5, %4
  %7 = phi [2 x i8]* [ getelementptr inbounds ([64 x [2 x i8]], [64 x [2 x i8]]* @FIELD_SCAN8x8, i32 0, i32 0), %4 ], [ null, %5 ]
  br i1 undef, label %switch.lookup6, label %8

switch.lookup6:                                   ; preds = %6
  br label %8

; <label>:8                                       ; preds = %switch.lookup6, %6
  br i1 undef, label %.loopexit, label %9

; <label>:9                                       ; preds = %8
  %10 = and i32 %b8, 1
  %11 = shl nuw nsw i32 %10, 3
  %12 = getelementptr inbounds %struct.Slice, %struct.Slice* null, i32 0, i32 9
  br i1 undef, label %.preheader, label %.preheader11

.preheader11:                                     ; preds = %21, %9
  %k.014 = phi i32 [ %27, %21 ], [ 0, %9 ]
  %coef_ctr.013 = phi i32 [ %23, %21 ], [ -1, %9 ]
  br i1 false, label %13, label %14

; <label>:13                                      ; preds = %.preheader11
  br label %15

; <label>:14                                      ; preds = %.preheader11
  br label %15

; <label>:15                                      ; preds = %14, %13
  %16 = getelementptr inbounds [0 x [20 x i32]], [0 x [20 x i32]]* @assignSE2partition, i32 0, i32 %1, i32 undef
  %17 = load i32, i32* %16, align 4
  %18 = getelementptr inbounds %struct.datapartition, %struct.datapartition* null, i32 %17, i32 2
  %19 = load i32 (%struct.syntaxelement*, %struct.img_par*, %struct.datapartition*)*, i32 (%struct.syntaxelement*, %struct.img_par*, %struct.datapartition*)** %18, align 4
  %20 = call i32 %19(%struct.syntaxelement* undef, %struct.img_par* %img, %struct.datapartition* undef)
  br i1 false, label %.loopexit, label %21

; <label>:21                                      ; preds = %15
  %22 = add i32 %coef_ctr.013, 1
  %23 = add i32 %22, 0
  %24 = getelementptr inbounds [2 x i8], [2 x i8]* %7, i32 %23, i32 0
  %25 = add nsw i32 0, %11
  %26 = getelementptr inbounds %struct.img_par, %struct.img_par* %img, i32 0, i32 27, i32 undef, i32 %25
  store i32 0, i32* %26, align 4
  %27 = add nsw i32 %k.014, 1
  %28 = icmp slt i32 %27, 65
  br i1 %28, label %.preheader11, label %.loopexit

.preheader:                                       ; preds = %36, %9
  %k.110 = phi i32 [ %45, %36 ], [ 0, %9 ]
  %coef_ctr.29 = phi i32 [ %39, %36 ], [ -1, %9 ]
  br i1 false, label %29, label %30

; <label>:29                                      ; preds = %.preheader
  br label %31

; <label>:30                                      ; preds = %.preheader
  br label %31

; <label>:31                                      ; preds = %30, %29
  %32 = getelementptr inbounds [0 x [20 x i32]], [0 x [20 x i32]]* @assignSE2partition, i32 0, i32 %1, i32 undef
  %33 = load i32, i32* %32, align 4
  %34 = getelementptr inbounds %struct.datapartition, %struct.datapartition* null, i32 %33
  %35 = call i32 undef(%struct.syntaxelement* undef, %struct.img_par* %img, %struct.datapartition* %34)
  br i1 false, label %.loopexit, label %36

; <label>:36                                      ; preds = %31
  %37 = load i32, i32* undef, align 4
  %38 = add i32 %coef_ctr.29, 1
  %39 = add i32 %38, %37
  %40 = getelementptr inbounds [2 x i8], [2 x i8]* %7, i32 %39, i32 0
  %41 = load i8, i8* %40, align 1
  %42 = zext i8 %41 to i32
  %43 = add nsw i32 %42, %11
  %44 = getelementptr inbounds %struct.img_par, %struct.img_par* %img, i32 0, i32 27, i32 undef, i32 %43
  store i32 0, i32* %44, align 4
  %45 = add nsw i32 %k.110, 1
  %46 = icmp slt i32 %45, 65
  br i1 %46, label %.preheader, label %.loopexit

.loopexit:                                        ; preds = %36, %31, %21, %15, %8
  ret void
}
