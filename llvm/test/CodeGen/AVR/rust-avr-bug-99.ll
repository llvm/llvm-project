; RUN: llc < %s -mtriple=avr -mcpu=avr5 | FileCheck %s

; The original reason for this failure is that the BranchFolderPass disables liveness
; tracking unless you override the trackLivenessAfterRegAlloc function and return true.
; This probably should be the default because all main targets do this (maybe some gpu targets don't).

; More info can be found at https://github.com/avr-rust/rust/issues/99.

%struct.quux = type { [0 x i8], i64, [0 x i8], i64, [0 x i8], i64, [0 x i8], i64, [0 x i8] }
%struct.foo = type { [0 x i8], %struct.blam, [0 x i8], i32, [0 x i8], i32, [0 x i8], i8, [0 x i8], %struct.blam.0, [0 x i8], %struct.blam.0, [0 x i8] }
%struct.blam = type {}
%struct.blam.0 = type { [0 x i8], i8, [2 x i8] }
%struct.quux.1 = type { [0 x i8], %struct.wombat, [0 x i8], i64, [0 x i8], i64, [0 x i8], i16, [0 x i8], %struct.quux, [0 x i8], i64, [0 x i8], i16, [0 x i8] }
%struct.wombat = type {}

declare zeroext i1 @zot(ptr, ptr)

declare void @wibble(i16, i16)

; CHECK-LABEL: main
define zeroext i1 @main(ptr %arg, ptr %arg62) {
bb:
  %tmp63 = alloca [128 x i8], align 1
  %tmp = getelementptr inbounds %struct.quux.1, ptr %arg, i16 0, i32 5
  %tmp64 = getelementptr inbounds %struct.quux.1, ptr %arg, i16 0, i32 13
  %tmp65 = bitcast ptr %arg62 to ptr
  %tmp66 = icmp eq i32 undef, 0
  br i1 undef, label %bb92, label %bb67

bb67:
  br i1 %tmp66, label %bb83, label %bb68

bb68:
  %tmp69 = load i64, ptr null, align 1
  br label %bb70

bb70:
  %tmp71 = phi i16 [ 128, %bb68 ], [ %tmp79, %bb70 ]
  %tmp72 = phi i64 [ %tmp69, %bb68 ], [ %tmp74, %bb70 ]
  %tmp73 = getelementptr inbounds i8, ptr null, i16 -1
  %tmp74 = lshr i64 %tmp72, 4
  %tmp75 = trunc i64 %tmp72 to i8
  %tmp76 = and i8 %tmp75, 15
  %tmp77 = add nuw nsw i8 %tmp76, 87
  %tmp78 = select i1 undef, i8 undef, i8 %tmp77
  store i8 %tmp78, ptr %tmp73, align 1
  %tmp79 = add nsw i16 %tmp71, -1
  %tmp80 = icmp eq ptr %tmp73, null
  %tmp81 = or i1 undef, %tmp80
  br i1 %tmp81, label %bb82, label %bb70

bb82:
  call void @wibble(i16 %tmp79, i16 128)
  unreachable

bb83:
  %tmp84 = icmp eq i32 undef, 0
  %tmp85 = load i64, ptr null, align 1
  br i1 %tmp84, label %bb87, label %bb86

bb86:
  unreachable

bb87:
  br label %bb88

bb88:
  %tmp89 = phi i64 [ %tmp90, %bb88 ], [ %tmp85, %bb87 ]
  %tmp90 = udiv i64 %tmp89, 10000
  %tmp91 = icmp ugt i64 %tmp89, 99999999
  br label %bb88

bb92:
  br label %bb93

bb93:
  br i1 undef, label %bb95, label %bb94

bb94:
  unreachable

bb95:
  br label %bb96

bb96:
  %tmp97 = phi i64 [ %tmp98, %bb96 ], [ undef, %bb95 ]
  %tmp98 = udiv i64 %tmp97, 10000
  %tmp99 = icmp ugt i64 %tmp97, 99999999
  br i1 %tmp99, label %bb96, label %bb100

bb100:
  br label %bb101

bb101:
  %tmp102 = and i32 undef, 16
  %tmp103 = icmp eq i32 %tmp102, 0
  br i1 undef, label %bb130, label %bb104

bb104:
  br i1 %tmp103, label %bb117, label %bb105

bb105:
  br label %bb106

bb106:
  %tmp107 = phi i16 [ 128, %bb105 ], [ %tmp113, %bb106 ]
  %tmp108 = phi i64 [ undef, %bb105 ], [ %tmp111, %bb106 ]
  %tmp109 = phi ptr [ undef, %bb105 ], [ %tmp110, %bb106 ]
  %tmp110 = getelementptr inbounds i8, ptr %tmp109, i16 -1
  %tmp111 = lshr i64 %tmp108, 4
  %tmp112 = trunc i64 %tmp108 to i8
  %tmp113 = add nsw i16 %tmp107, -1
  %tmp114 = icmp eq ptr %tmp110, null
  %tmp115 = or i1 undef, %tmp114
  br i1 %tmp115, label %bb116, label %bb106

bb116:
  call void @wibble(i16 %tmp113, i16 128)
  unreachable

bb117:
  %tmp118 = load i64, ptr %tmp, align 1
  br i1 undef, label %bb120, label %bb119

bb119:
  unreachable

bb120:
  %tmp121 = icmp ugt i64 %tmp118, 9999
  br i1 %tmp121, label %bb122, label %bb127

bb122:
  br label %bb123

bb123:
  %tmp124 = phi i64 [ %tmp125, %bb123 ], [ %tmp118, %bb122 ]
  %tmp125 = udiv i64 %tmp124, 10000
  %tmp126 = icmp ugt i64 %tmp124, 99999999
  br label %bb123

bb127:
  %tmp128 = load i32, ptr %tmp65, align 1
  %tmp129 = icmp eq i32 undef, 0
  br label %bb162

bb130:
  br i1 %tmp103, label %bb142, label %bb131

bb131:
  br label %bb132

bb132:
  %tmp133 = phi i64 [ undef, %bb131 ], [ %tmp134, %bb132 ]
  %tmp134 = lshr i64 %tmp133, 4
  %tmp135 = trunc i64 %tmp133 to i8
  %tmp136 = and i8 %tmp135, 15
  %tmp137 = add nuw nsw i8 %tmp136, 87
  %tmp138 = select i1 undef, i8 undef, i8 %tmp137
  store i8 %tmp138, ptr undef, align 1
  %tmp139 = icmp eq ptr undef, null
  %tmp140 = or i1 undef, %tmp139
  br i1 %tmp140, label %bb141, label %bb132

bb141:
  unreachable

bb142:
  %tmp143 = icmp eq i32 undef, 0
  %tmp144 = load i64, ptr %tmp, align 1
  br i1 %tmp143, label %bb156, label %bb145

bb145:
  br label %bb146

bb146:
  %tmp147 = phi i16 [ 128, %bb145 ], [ %tmp151, %bb146 ]
  %tmp148 = phi i64 [ %tmp144, %bb145 ], [ %tmp150, %bb146 ]
  %tmp149 = getelementptr inbounds i8, ptr null, i16 -1
  %tmp150 = lshr i64 %tmp148, 4
  %tmp151 = add nsw i16 %tmp147, -1
  %tmp152 = icmp eq i64 %tmp150, 0
  %tmp153 = icmp eq ptr %tmp149, null
  %tmp154 = or i1 %tmp152, %tmp153
  br i1 %tmp154, label %bb155, label %bb146

bb155:
  call void @wibble(i16 %tmp151, i16 128)
  unreachable

bb156:
  br label %bb157

bb157:
  %tmp158 = phi i64 [ %tmp159, %bb157 ], [ %tmp144, %bb156 ]
  %tmp159 = udiv i64 %tmp158, 10000
  %tmp160 = icmp ugt i64 %tmp158, 99999999
  br i1 %tmp160, label %bb157, label %bb161

bb161:
  unreachable

bb162:
  br i1 %tmp129, label %bb164, label %bb163

bb163:
  unreachable

bb164:
  %tmp165 = and i32 %tmp128, 32
  %tmp166 = icmp eq i32 %tmp165, 0
  br i1 %tmp166, label %bb169, label %bb167

bb167:
  br label %bb168

bb168:
  br label %bb168

bb169:
  br label %bb170

bb170:
  br i1 undef, label %bb172, label %bb171

bb171:
  store i32 0, ptr undef, align 1
  call void @llvm.memcpy.p0.p0.i16(ptr align 1 undef, ptr align 1 null, i16 3, i1 false)
  call void @llvm.memcpy.p0.p0.i16(ptr align 1 undef, ptr align 1 null, i16 3, i1 false)
  br label %bb214

bb172:
  %tmp173 = call zeroext i1 @zot(ptr noalias nonnull readonly dereferenceable(32) undef, ptr nonnull dereferenceable(15) %arg62)
  br i1 %tmp173, label %bb214, label %bb174

bb174:
  %tmp175 = load i32, ptr %tmp65, align 1
  %tmp176 = icmp eq i32 undef, 0
  br label %bb177

bb177:
  br i1 %tmp176, label %bb190, label %bb178

bb178:
  %tmp179 = getelementptr inbounds [128 x i8], ptr %tmp63, i16 0, i16 0
  br label %bb180

bb180:
  %tmp181 = phi i64 [ 0, %bb178 ], [ %tmp182, %bb180 ]
  %tmp182 = lshr i64 %tmp181, 4
  %tmp183 = trunc i64 %tmp181 to i8
  %tmp184 = and i8 %tmp183, 15
  %tmp185 = add nuw nsw i8 %tmp184, 87
  %tmp186 = select i1 false, i8 0, i8 %tmp185
  store i8 %tmp186, ptr null, align 1
  %tmp187 = icmp eq ptr null, %tmp179
  %tmp188 = or i1 undef, %tmp187
  br i1 %tmp188, label %bb189, label %bb180

bb189:
  call void @wibble(i16 0, i16 128)
  unreachable

bb190:
  %tmp191 = and i32 %tmp175, 32
  %tmp192 = icmp eq i32 %tmp191, 0
  br i1 %tmp192, label %bb201, label %bb193

bb193:
  br label %bb194

bb194:
  %tmp195 = phi i64 [ 0, %bb193 ], [ %tmp196, %bb194 ]
  %tmp196 = lshr i64 %tmp195, 4
  %tmp197 = add nsw i16 0, -1
  %tmp198 = icmp eq i64 %tmp196, 0
  %tmp199 = or i1 %tmp198, undef
  br i1 %tmp199, label %bb200, label %bb194

bb200:
  call void @wibble(i16 %tmp197, i16 128)
  unreachable

bb201:
  br i1 undef, label %bb202, label %bb207

bb202:
  br label %bb203

bb203:
  %tmp204 = phi i64 [ %tmp205, %bb203 ], [ 0, %bb202 ]
  %tmp205 = udiv i64 %tmp204, 10000
  %tmp206 = icmp ugt i64 %tmp204, 99999999
  br i1 %tmp206, label %bb203, label %bb207

bb207:
  br label %bb208

bb208:
  store ptr %tmp64, ptr undef, align 1
  %tmp209 = load i32, ptr %tmp65, align 1
  %tmp210 = icmp eq i32 undef, 0
  %tmp211 = and i32 %tmp209, 16
  %tmp212 = icmp eq i32 %tmp211, 0
  br i1 %tmp210, label %bb215, label %bb213

bb213:
  unreachable

bb214:
  br label %bb221

bb215:
  br i1 %tmp212, label %bb220, label %bb216

bb216:
  br label %bb217

bb217:
  br i1 undef, label %bb218, label %bb219

bb218:
  unreachable

bb219:
  br label %bb221

bb220:
  unreachable

bb221:
  store ptr %arg, ptr undef, align 1
  ret i1 undef
}

declare void @llvm.memcpy.p0.p0.i16(ptr nocapture writeonly, ptr nocapture readonly, i16, i1)

