; RUN: opt -basic-aa -aa -memoryssa -early-cse-memssa -verify -S < %s | FileCheck %s

; Check that this doesn't crash. The crash only happens with expensive checks,
; but there doesn't seem to be a REQUIRES for that.

; CHECK: invoke i32 @f10

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%s.90 = type { %s.91 }
%s.91 = type { %s.92* }
%s.92 = type { %s.93, %s.96, %s.83 }
%s.93 = type { %s.94, %s.95 }
%s.94 = type { i32 (...)**, i64 }
%s.95 = type { i32 (...)** }
%s.96 = type <{ %s.97, %s.0, i8*, i32, [4 x i8] }>
%s.97 = type { i32 (...)**, %s.98, i8*, i8*, i8*, i8*, i8*, i8* }
%s.98 = type { %s.99* }
%s.99 = type opaque
%s.0 = type { %s.1 }
%s.1 = type { %s.2 }
%s.2 = type { %s.3 }
%s.3 = type { %s.4 }
%s.4 = type { %s.5 }
%s.5 = type { i64, i64, i8* }
%s.83 = type <{ %s.84, %s.82*, i32 }>
%s.84 = type { i32 (...)**, i32, i64, i64, i32, i32, i8*, i8*, void (i32, %s.84*, i32)**, i32*, i64, i64, i64*, i64, i64, i8**, i64, i64 }
%s.82 = type { i32 (...)**, %s.83 }
%s.161 = type { i8, %s.162 }
%s.162 = type { %s.163 }
%s.163 = type { %s.164*, %s.166, %s.168 }
%s.164 = type { %s.165* }
%s.165 = type <{ %s.164, %s.165*, %s.164*, i8, [7 x i8] }>
%s.166 = type { %s.167 }
%s.167 = type { %s.164 }
%s.168 = type { %s.169 }
%s.169 = type { i64 }
%s.10 = type { %s.11 }
%s.11 = type { %s.0*, %s.0*, %s.12 }
%s.12 = type { %s.13 }
%s.13 = type { %s.0* }
%s.170 = type { %s.171 }
%s.171 = type { %s.164*, %s.172, %s.173 }
%s.172 = type { %s.167 }
%s.173 = type { %s.169 }

@g0 = external dso_local unnamed_addr constant [1 x i8], align 1
@g1 = external dso_local unnamed_addr constant [3 x i8], align 1
@g2 = external dso_local unnamed_addr constant [6 x i8], align 1
@g3 = external dso_local unnamed_addr constant [28 x i8], align 1
@g4 = external dso_local unnamed_addr constant [15 x i8], align 1
@g5 = external dso_local unnamed_addr constant [34 x i8], align 1
@g6 = external dso_local unnamed_addr constant [25 x i8], align 1

declare dso_local i32 @f0(...)

; Function Attrs: uwtable
declare dso_local void @f1(%s.90* nocapture) unnamed_addr #0 align 2

declare dso_local void @f2(%s.0*, %s.92*) local_unnamed_addr #1

declare dso_local void @f3(%s.0*, i8*, i32) local_unnamed_addr #1

define dso_local i8* @f4(%s.161* %a0, i8* %a1, i32 %a2, i8* %a3) local_unnamed_addr #1 align 2 personality i8* bitcast (i32 (...)* @f0 to i8*) {
b0:
  %v0 = alloca %s.90, align 8
  br label %b1

b1:                                               ; preds = %b2, %b0
  br i1 undef, label %b2, label %b3

b2:                                               ; preds = %b1
  br label %b1

b3:                                               ; preds = %b1
  br label %b4

b4:                                               ; preds = %b40, %b3
  br i1 undef, label %b57, label %b5

b5:                                               ; preds = %b4
  invoke void @f7(%s.0* nonnull sret align 8 undef, i8* nonnull undef)
          to label %b6 unwind label %b41

b6:                                               ; preds = %b5
  br i1 undef, label %b7, label %b8

b7:                                               ; preds = %b6
  br label %b9

b8:                                               ; preds = %b6
  br label %b9

b9:                                               ; preds = %b8, %b7
  br i1 undef, label %b10, label %b11

b10:                                              ; preds = %b9
  br label %b12

b11:                                              ; preds = %b9
  br label %b12

b12:                                              ; preds = %b11, %b10
  br label %b13

b13:                                              ; preds = %b22, %b12
  br i1 undef, label %b14, label %b15

b14:                                              ; preds = %b13
  br label %b16

b15:                                              ; preds = %b13
  br label %b16

b16:                                              ; preds = %b15, %b14
  br i1 undef, label %b17, label %b23

b17:                                              ; preds = %b16
  br i1 undef, label %b18, label %b24

b18:                                              ; preds = %b17
  br i1 undef, label %b19, label %b20

b19:                                              ; preds = %b18
  br label %b21

b20:                                              ; preds = %b18
  br label %b21

b21:                                              ; preds = %b20, %b19
  %v1 = invoke nonnull align 8 dereferenceable(24) %s.0* @f8(%s.0* undef, i64 undef, i64 1)
          to label %b22 unwind label %b42

b22:                                              ; preds = %b21
  br label %b13

b23:                                              ; preds = %b16
  br label %b24

b24:                                              ; preds = %b23, %b17
  br label %b25

b25:                                              ; preds = %b26, %b24
  br i1 undef, label %b26, label %b27

b26:                                              ; preds = %b25
  br label %b25

b27:                                              ; preds = %b25
  br i1 undef, label %b28, label %b32

b28:                                              ; preds = %b27
  br label %b29

b29:                                              ; preds = %b30, %b28
  br i1 undef, label %b30, label %b31

b30:                                              ; preds = %b29
  br label %b29

b31:                                              ; preds = %b29
  br label %b34

b32:                                              ; preds = %b27
  invoke void @f9(%s.10* undef, %s.0* nonnull align 8 dereferenceable(24) undef)
          to label %b33 unwind label %b43

b33:                                              ; preds = %b32
  br label %b34

b34:                                              ; preds = %b33, %b31
  br i1 undef, label %b35, label %b36

b35:                                              ; preds = %b34
  br label %b36

b36:                                              ; preds = %b35, %b34
  br i1 undef, label %b37, label %b38

b37:                                              ; preds = %b36
  br label %b38

b38:                                              ; preds = %b37, %b36
  br i1 undef, label %b40, label %b39

b39:                                              ; preds = %b39, %b38
  br i1 undef, label %b39, label %b40

b40:                                              ; preds = %b39, %b38
  br label %b4

b41:                                              ; preds = %b5
  %v2 = landingpad { i8*, i32 }
          cleanup
  br label %b49

b42:                                              ; preds = %b21
  %v3 = landingpad { i8*, i32 }
          cleanup
  br label %b46

b43:                                              ; preds = %b32
  %v4 = landingpad { i8*, i32 }
          cleanup
  br i1 undef, label %b44, label %b45

b44:                                              ; preds = %b43
  br label %b45

b45:                                              ; preds = %b44, %b43
  br label %b46

b46:                                              ; preds = %b45, %b42
  br i1 undef, label %b47, label %b48

b47:                                              ; preds = %b46
  br label %b48

b48:                                              ; preds = %b47, %b46
  br label %b49

b49:                                              ; preds = %b48, %b41
  br i1 undef, label %b56, label %b50

b50:                                              ; preds = %b49
  br label %b51

b51:                                              ; preds = %b54, %b50
  br i1 undef, label %b55, label %b52

b52:                                              ; preds = %b51
  br i1 undef, label %b53, label %b54

b53:                                              ; preds = %b52
  br label %b54

b54:                                              ; preds = %b53, %b52
  br label %b51

b55:                                              ; preds = %b51
  br label %b56

b56:                                              ; preds = %b55, %b49
  resume { i8*, i32 } undef

b57:                                              ; preds = %b4
  invoke void @f1(%s.90* nonnull %v0)
          to label %b58 unwind label %b61

b58:                                              ; preds = %b57
  br label %b59

b59:                                              ; preds = %b130, %b58
  br i1 undef, label %b62, label %b60

b60:                                              ; preds = %b59
  br label %b132

b61:                                              ; preds = %b57
  %v5 = landingpad { i8*, i32 }
          cleanup
  br label %b205

b62:                                              ; preds = %b59
  %v6 = invoke i64 @f5(%s.170* nonnull undef, %s.0* nonnull align 8 dereferenceable(24) undef)
          to label %b63 unwind label %b76

b63:                                              ; preds = %b62
  br i1 undef, label %b77, label %b64

b64:                                              ; preds = %b63
  %v7 = invoke nonnull align 8 dereferenceable(8) %s.82* @f6(%s.82* nonnull align 8 dereferenceable(8) undef, i8* getelementptr inbounds ([6 x i8], [6 x i8]* @g2, i64 0, i64 0), i64 undef)
          to label %b65 unwind label %b76

b65:                                              ; preds = %b64
  br label %b66

b66:                                              ; preds = %b65
  br i1 undef, label %b67, label %b68

b67:                                              ; preds = %b66
  br label %b69

b68:                                              ; preds = %b66
  br label %b69

b69:                                              ; preds = %b68, %b67
  br i1 undef, label %b70, label %b71

b70:                                              ; preds = %b69
  br label %b72

b71:                                              ; preds = %b69
  br label %b72

b72:                                              ; preds = %b71, %b70
  %v8 = invoke nonnull align 8 dereferenceable(8) %s.82* @f6(%s.82* nonnull align 8 dereferenceable(8) undef, i8* undef, i64 undef)
          to label %b73 unwind label %b76

b73:                                              ; preds = %b72
  br label %b74

b74:                                              ; preds = %b73
  %v9 = invoke nonnull align 8 dereferenceable(8) %s.82* @f6(%s.82* nonnull align 8 dereferenceable(8) undef, i8* getelementptr inbounds ([28 x i8], [28 x i8]* @g3, i64 0, i64 0), i64 undef)
          to label %b75 unwind label %b76

b75:                                              ; preds = %b74
  br label %b130

b76:                                              ; preds = %b74, %b72, %b64, %b62
  %v10 = landingpad { i8*, i32 }
          cleanup
  br label %b131

b77:                                              ; preds = %b63
  br label %b78

b78:                                              ; preds = %b111, %b77
  br label %b79

b79:                                              ; preds = %b78
  br i1 undef, label %b81, label %b117

b80:                                              ; No predecessors!
  %v11 = landingpad { i8*, i32 }
          cleanup
  br label %b113

b81:                                              ; preds = %b79
  br label %b82

b82:                                              ; preds = %b81
  br i1 undef, label %b83, label %b84

b83:                                              ; preds = %b82
  br label %b85

b84:                                              ; preds = %b82
  br label %b85

b85:                                              ; preds = %b84, %b83
  br i1 undef, label %b86, label %b87

b86:                                              ; preds = %b85
  br label %b88

b87:                                              ; preds = %b85
  br label %b88

b88:                                              ; preds = %b87, %b86
  br i1 undef, label %b89, label %b102

b89:                                              ; preds = %b88
  br i1 undef, label %b90, label %b91

b90:                                              ; preds = %b89
  br label %b92

b91:                                              ; preds = %b89
  br label %b92

b92:                                              ; preds = %b91, %b90
  br i1 undef, label %b93, label %b94

b93:                                              ; preds = %b92
  br label %b95

b94:                                              ; preds = %b92
  br label %b95

b95:                                              ; preds = %b94, %b93
  br i1 undef, label %b96, label %b99

b96:                                              ; preds = %b95
  br i1 undef, label %b98, label %b97

b97:                                              ; preds = %b96
  br label %b98

b98:                                              ; preds = %b97, %b96
  br label %b102

b99:                                              ; preds = %b101, %b95
  br i1 undef, label %b102, label %b100

b100:                                             ; preds = %b99
  br i1 undef, label %b101, label %b102

b101:                                             ; preds = %b100
  br label %b99

b102:                                             ; preds = %b100, %b99, %b98, %b88
  br i1 undef, label %b112, label %b104

b103:                                             ; No predecessors!
  %v12 = landingpad { i8*, i32 }
          cleanup
  br label %b113

b104:                                             ; preds = %b102
  br i1 undef, label %b108, label %b105

b105:                                             ; preds = %b104
  br label %b106

b106:                                             ; preds = %b106, %b105
  br i1 undef, label %b107, label %b106

b107:                                             ; preds = %b106
  br label %b111

b108:                                             ; preds = %b109, %b104
  br i1 undef, label %b110, label %b109

b109:                                             ; preds = %b108
  br label %b108

b110:                                             ; preds = %b108
  br label %b111

b111:                                             ; preds = %b110, %b107
  br label %b78

b112:                                             ; preds = %b102
  br i1 undef, label %b114, label %b118

b113:                                             ; preds = %b103, %b80
  br label %b131

b114:                                             ; preds = %b112
  %v13 = invoke { %s.164*, i8 } @f11(%s.171* undef, %s.0* nonnull align 8 dereferenceable(24) undef, %s.0* nonnull align 8 dereferenceable(24) undef)
          to label %b115 unwind label %b116

b115:                                             ; preds = %b114
  br label %b130

b116:                                             ; preds = %b128, %b126, %b118, %b114
  %v14 = landingpad { i8*, i32 }
          cleanup
  br label %b131

b117:                                             ; preds = %b79
  br label %b118

b118:                                             ; preds = %b117, %b112
  %v15 = invoke nonnull align 8 dereferenceable(8) %s.82* @f6(%s.82* nonnull align 8 dereferenceable(8) undef, i8* getelementptr inbounds ([15 x i8], [15 x i8]* @g4, i64 0, i64 0), i64 undef)
          to label %b119 unwind label %b116

b119:                                             ; preds = %b118
  br label %b120

b120:                                             ; preds = %b119
  br i1 undef, label %b121, label %b122

b121:                                             ; preds = %b120
  br label %b123

b122:                                             ; preds = %b120
  br label %b123

b123:                                             ; preds = %b122, %b121
  br i1 undef, label %b124, label %b125

b124:                                             ; preds = %b123
  br label %b126

b125:                                             ; preds = %b123
  br label %b126

b126:                                             ; preds = %b125, %b124
  %v16 = invoke nonnull align 8 dereferenceable(8) %s.82* @f6(%s.82* nonnull align 8 dereferenceable(8) undef, i8* undef, i64 undef)
          to label %b127 unwind label %b116

b127:                                             ; preds = %b126
  br label %b128

b128:                                             ; preds = %b127
  %v17 = invoke nonnull align 8 dereferenceable(8) %s.82* @f6(%s.82* nonnull align 8 dereferenceable(8) undef, i8* getelementptr inbounds ([34 x i8], [34 x i8]* @g5, i64 0, i64 0), i64 undef)
          to label %b129 unwind label %b116

b129:                                             ; preds = %b128
  br label %b130

b130:                                             ; preds = %b129, %b115, %b75
  br label %b59

b131:                                             ; preds = %b116, %b113, %b76
  br label %b200

b132:                                             ; preds = %b161, %b60
  br label %b133

b133:                                             ; preds = %b132
  br i1 undef, label %b137, label %b134

b134:                                             ; preds = %b133
  invoke void @f2(%s.0* sret align 8 undef, %s.92* undef)
          to label %b135 unwind label %b182

b135:                                             ; preds = %b134
  br label %b163

b136:                                             ; No predecessors!
  %v18 = landingpad { i8*, i32 }
          cleanup
  br label %b162

b137:                                             ; preds = %b133
  br label %b138

b138:                                             ; preds = %b137
  %v19 = invoke i64 @f5(%s.170* nonnull undef, %s.0* nonnull align 8 dereferenceable(24) undef)
          to label %b139 unwind label %b153

b139:                                             ; preds = %b138
  br i1 undef, label %b140, label %b154

b140:                                             ; preds = %b139
  %v20 = invoke nonnull align 8 dereferenceable(8) %s.82* @f6(%s.82* nonnull align 8 dereferenceable(8) undef, i8* getelementptr inbounds ([25 x i8], [25 x i8]* @g6, i64 0, i64 0), i64 undef)
          to label %b141 unwind label %b153

b141:                                             ; preds = %b140
  br label %b142

b142:                                             ; preds = %b141
  br label %b143

b143:                                             ; preds = %b142
  br i1 undef, label %b144, label %b145

b144:                                             ; preds = %b143
  br label %b146

b145:                                             ; preds = %b143
  br label %b146

b146:                                             ; preds = %b145, %b144
  br i1 undef, label %b147, label %b148

b147:                                             ; preds = %b146
  br label %b149

b148:                                             ; preds = %b146
  br label %b149

b149:                                             ; preds = %b148, %b147
  %v21 = invoke nonnull align 8 dereferenceable(8) %s.82* @f6(%s.82* nonnull align 8 dereferenceable(8) undef, i8* undef, i64 undef)
          to label %b150 unwind label %b153

b150:                                             ; preds = %b149
  br label %b151

b151:                                             ; preds = %b150
  %v22 = invoke nonnull align 8 dereferenceable(8) %s.82* @f6(%s.82* nonnull align 8 dereferenceable(8) undef, i8* getelementptr inbounds ([3 x i8], [3 x i8]* @g1, i64 0, i64 0), i64 undef)
          to label %b152 unwind label %b153

b152:                                             ; preds = %b151
  br label %b154

b153:                                             ; preds = %b151, %b149, %b140, %b138
  %v23 = landingpad { i8*, i32 }
          cleanup
  br label %b162

b154:                                             ; preds = %b152, %b139
  br i1 undef, label %b158, label %b155

b155:                                             ; preds = %b154
  br label %b156

b156:                                             ; preds = %b156, %b155
  br i1 undef, label %b157, label %b156

b157:                                             ; preds = %b156
  br label %b161

b158:                                             ; preds = %b159, %b154
  br i1 undef, label %b160, label %b159

b159:                                             ; preds = %b158
  br label %b158

b160:                                             ; preds = %b158
  br label %b161

b161:                                             ; preds = %b160, %b157
  br label %b132

b162:                                             ; preds = %b153, %b136
  br label %b200

b163:                                             ; preds = %b135
  br i1 undef, label %b164, label %b165

b164:                                             ; preds = %b163
  br label %b166

b165:                                             ; preds = %b163
  br label %b166

b166:                                             ; preds = %b165, %b164
  br i1 undef, label %b167, label %b170

b167:                                             ; preds = %b166
  %v24 = invoke i32 @f10(%s.0* nonnull undef, i64 0, i64 -1, i8* getelementptr inbounds ([1 x i8], [1 x i8]* @g0, i64 0, i64 0), i64 undef)
          to label %b168 unwind label %b169

b168:                                             ; preds = %b167
  br label %b170

b169:                                             ; preds = %b167
  %v25 = landingpad { i8*, i32 }
          catch i8* null
  unreachable

b170:                                             ; preds = %b168, %b166
  br i1 undef, label %b171, label %b186

b171:                                             ; preds = %b170
  invoke void @f3(%s.0* nonnull sret align 8 undef, i8* %a1, i32 %a2)
          to label %b172 unwind label %b183

b172:                                             ; preds = %b171
  br i1 undef, label %b173, label %b174

b173:                                             ; preds = %b172
  br label %b175

b174:                                             ; preds = %b172
  br label %b175

b175:                                             ; preds = %b174, %b173
  br i1 undef, label %b176, label %b177

b176:                                             ; preds = %b175
  br label %b178

b177:                                             ; preds = %b175
  br label %b178

b178:                                             ; preds = %b177, %b176
  br i1 undef, label %b179, label %b180

b179:                                             ; preds = %b178
  br label %b180

b180:                                             ; preds = %b179, %b178
  unreachable

b181:                                             ; No predecessors!
  br label %b186

b182:                                             ; preds = %b134
  %v26 = landingpad { i8*, i32 }
          cleanup
  br label %b200

b183:                                             ; preds = %b171
  %v27 = landingpad { i8*, i32 }
          cleanup
  br i1 undef, label %b184, label %b185

b184:                                             ; preds = %b183
  br label %b185

b185:                                             ; preds = %b184, %b183
  br label %b200

b186:                                             ; preds = %b181, %b170
  br i1 undef, label %b187, label %b188

b187:                                             ; preds = %b186
  br label %b188

b188:                                             ; preds = %b187, %b186
  br i1 undef, label %b192, label %b189

b189:                                             ; preds = %b188
  br i1 undef, label %b191, label %b190

b190:                                             ; preds = %b189
  br label %b191

b191:                                             ; preds = %b190, %b189
  br label %b192

b192:                                             ; preds = %b191, %b188
  br i1 undef, label %b199, label %b193

b193:                                             ; preds = %b192
  br label %b194

b194:                                             ; preds = %b197, %b193
  br i1 undef, label %b198, label %b195

b195:                                             ; preds = %b194
  br i1 undef, label %b196, label %b197

b196:                                             ; preds = %b195
  br label %b197

b197:                                             ; preds = %b196, %b195
  br label %b194

b198:                                             ; preds = %b194
  br label %b199

b199:                                             ; preds = %b198, %b192
  ret i8* %a3

b200:                                             ; preds = %b185, %b182, %b162, %b131
  %v28 = getelementptr inbounds %s.90, %s.90* %v0, i64 0, i32 0
  %v29 = getelementptr inbounds %s.91, %s.91* %v28, i64 0, i32 0
  br i1 undef, label %b204, label %b201

b201:                                             ; preds = %b200
  %v30 = load %s.92*, %s.92** %v29, align 8
  br i1 undef, label %b203, label %b202

b202:                                             ; preds = %b201
  call void undef(%s.92* nonnull %v30) #2
  br label %b203

b203:                                             ; preds = %b202, %b201
  store %s.92* null, %s.92** %v29, align 8
  br label %b204

b204:                                             ; preds = %b203, %b200
  br label %b205

b205:                                             ; preds = %b204, %b61
  br i1 undef, label %b212, label %b206

b206:                                             ; preds = %b205
  br label %b207

b207:                                             ; preds = %b210, %b206
  br i1 undef, label %b211, label %b208

b208:                                             ; preds = %b207
  br i1 undef, label %b209, label %b210

b209:                                             ; preds = %b208
  br label %b210

b210:                                             ; preds = %b209, %b208
  br label %b207

b211:                                             ; preds = %b207
  br label %b212

b212:                                             ; preds = %b211, %b205
  resume { i8*, i32 } undef
}

declare hidden i64 @f5(%s.170*, %s.0*) local_unnamed_addr #1 align 2

declare hidden %s.82* @f6(%s.82*, i8*, i64) local_unnamed_addr #1

declare hidden void @f7(%s.0*, i8*) local_unnamed_addr #1

declare dso_local %s.0* @f8(%s.0*, i64, i64) local_unnamed_addr #1

declare hidden void @f9(%s.10*, %s.0*) local_unnamed_addr #1 align 2

declare dso_local i32 @f10(%s.0*, i64, i64, i8*, i64) local_unnamed_addr #1

declare hidden { %s.164*, i8 } @f11(%s.171*, %s.0*, %s.0*) local_unnamed_addr #1 align 2

attributes #0 = { uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "use-soft-float"="false" }
attributes #2 = { nounwind }
