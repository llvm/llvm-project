; Ensure that we do not crash on functions with more than 256 basic blocks.
; RUN: opt -passes="bounds-checking<trap>" %s -disable-output

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-grtev4-linux-gnu"

define i8 @_ZNSt3__u18__d2exp_buffered_nEPcS0_dj() {
  br label %11

1:                                                ; No predecessors!
  br i1 false, label %2, label %3

2:                                                ; preds = %1
  unreachable

3:                                                ; preds = %1
  br i1 false, label %4, label %5

4:                                                ; preds = %3
  unreachable

5:                                                ; preds = %3
  br label %6

6:                                                ; preds = %5
  br i1 false, label %7, label %8

7:                                                ; preds = %196, %6
  unreachable

8:                                                ; preds = %6
  br label %9

9:                                                ; preds = %8
  store i8 0, ptr null, align 1
  br label %10

10:                                               ; preds = %9
  store i32 0, ptr null, align 1
  br label %209

11:                                               ; preds = %0
  br label %12

12:                                               ; preds = %11
  br i1 false, label %15, label %13

13:                                               ; preds = %12
  br i1 false, label %15, label %14

14:                                               ; preds = %13
  unreachable

15:                                               ; preds = %13, %12
  br i1 false, label %16, label %19

16:                                               ; preds = %15
  %17 = load i16, ptr null, align 2
  br label %18

18:                                               ; preds = %16
  br i1 false, label %20, label %21

19:                                               ; preds = %15
  unreachable

20:                                               ; preds = %18
  unreachable

21:                                               ; preds = %18
  br i1 false, label %23, label %22

22:                                               ; preds = %21
  unreachable

23:                                               ; preds = %21
  br i1 false, label %34, label %24

24:                                               ; preds = %23
  br label %25

25:                                               ; preds = %24
  br i1 false, label %209, label %26

26:                                               ; preds = %25
  br i1 false, label %27, label %28

27:                                               ; preds = %26
  br label %33

28:                                               ; preds = %26
  %29 = load i16, ptr null, align 1
  br label %30

30:                                               ; preds = %28
  br i1 false, label %210, label %31

31:                                               ; preds = %30
  br label %33

32:                                               ; No predecessors!
  unreachable

33:                                               ; preds = %31, %27
  br label %76

34:                                               ; preds = %23
  br label %35

35:                                               ; preds = %34
  br i1 false, label %37, label %36

36:                                               ; preds = %35
  unreachable

37:                                               ; preds = %35
  br label %38

38:                                               ; preds = %37
  br i1 false, label %45, label %39

39:                                               ; preds = %38
  br i1 false, label %45, label %40

40:                                               ; preds = %39
  br label %41

41:                                               ; preds = %40
  br i1 false, label %45, label %42

42:                                               ; preds = %41
  br i1 false, label %45, label %43

43:                                               ; preds = %42
  br label %44

44:                                               ; preds = %43
  br label %45

45:                                               ; preds = %44, %42, %41, %39, %38
  br i1 false, label %46, label %47

46:                                               ; preds = %45
  unreachable

47:                                               ; preds = %45
  br i1 false, label %48, label %49

48:                                               ; preds = %47
  unreachable

49:                                               ; preds = %47
  br i1 false, label %136, label %50

50:                                               ; preds = %49
  br i1 false, label %73, label %51

51:                                               ; preds = %50
  br label %52

52:                                               ; preds = %51
  br i1 false, label %53, label %61

53:                                               ; preds = %52
  br label %54

54:                                               ; preds = %53
  store i16 0, ptr null, align 1
  br i1 false, label %56, label %55

55:                                               ; preds = %54
  unreachable

56:                                               ; preds = %54
  br i1 false, label %70, label %57

57:                                               ; preds = %56
  br i1 false, label %211, label %58

58:                                               ; preds = %57
  %59 = load i16, ptr null, align 1
  store i16 0, ptr null, align 1
  br label %60

60:                                               ; preds = %58
  br label %61

61:                                               ; preds = %60, %52
  br label %62

62:                                               ; preds = %61
  %63 = load i16, ptr null, align 1
  store i16 0, ptr null, align 1
  br label %64

64:                                               ; preds = %62
  br i1 false, label %65, label %69

65:                                               ; preds = %64
  br i1 false, label %71, label %66

66:                                               ; preds = %65
  %67 = load i8, ptr null, align 1
  store i8 0, ptr null, align 1
  store i8 0, ptr null, align 1
  %68 = load i8, ptr null, align 1
  ret i8 0

69:                                               ; preds = %64
  store i8 0, ptr null, align 1
  br label %72

70:                                               ; preds = %56
  unreachable

71:                                               ; preds = %65
  unreachable

72:                                               ; preds = %69
  br label %76

73:                                               ; preds = %50
  br i1 false, label %209, label %74

74:                                               ; preds = %73
  br label %75

75:                                               ; preds = %74
  br label %76

76:                                               ; preds = %75, %72, %33
  br label %77

77:                                               ; preds = %76
  br i1 false, label %78, label %135

78:                                               ; preds = %77
  br i1 false, label %212, label %79

79:                                               ; preds = %78
  %80 = load i8, ptr null, align 1
  %81 = load i16, ptr null, align 2
  br label %82

82:                                               ; preds = %79
  %83 = load i16, ptr null, align 2
  br label %84

84:                                               ; preds = %134, %82
  br i1 false, label %85, label %88

85:                                               ; preds = %84
  br i1 false, label %87, label %86

86:                                               ; preds = %85
  unreachable

87:                                               ; preds = %85
  br i1 false, label %102, label %89

88:                                               ; preds = %84
  br label %134

89:                                               ; preds = %87
  br label %90

90:                                               ; preds = %89
  br i1 false, label %209, label %91

91:                                               ; preds = %90
  br i1 false, label %92, label %93

92:                                               ; preds = %91
  br label %101

93:                                               ; preds = %91
  %94 = load i16, ptr null, align 1
  store i16 0, ptr null, align 1
  br label %95

95:                                               ; preds = %93
  br i1 false, label %214, label %96

96:                                               ; preds = %95
  %97 = load i16, ptr null, align 1
  store i16 0, ptr null, align 1
  %98 = load i16, ptr null, align 1
  store i16 0, ptr null, align 1
  %99 = load i16, ptr null, align 1
  store i16 0, ptr null, align 1
  store i8 0, ptr null, align 1
  br label %101

100:                                              ; No predecessors!
  unreachable

101:                                              ; preds = %96, %92
  br label %134

102:                                              ; preds = %87
  br label %103

103:                                              ; preds = %102
  br i1 false, label %105, label %104

104:                                              ; preds = %103
  unreachable

105:                                              ; preds = %103
  br i1 false, label %113, label %106

106:                                              ; preds = %105
  br label %107

107:                                              ; preds = %106
  br i1 false, label %113, label %108

108:                                              ; preds = %107
  br label %109

109:                                              ; preds = %108
  br i1 false, label %113, label %110

110:                                              ; preds = %109
  br i1 false, label %113, label %111

111:                                              ; preds = %110
  br label %112

112:                                              ; preds = %111
  br label %113

113:                                              ; preds = %112, %110, %109, %107, %105
  br label %114

114:                                              ; preds = %113
  br i1 false, label %133, label %115

115:                                              ; preds = %114
  br label %116

116:                                              ; preds = %115
  br i1 false, label %117, label %124

117:                                              ; preds = %116
  br label %118

118:                                              ; preds = %117
  br i1 false, label %120, label %119

119:                                              ; preds = %118
  unreachable

120:                                              ; preds = %118
  br i1 false, label %130, label %121

121:                                              ; preds = %120
  br i1 false, label %215, label %122

122:                                              ; preds = %121
  br label %123

123:                                              ; preds = %122
  br label %124

124:                                              ; preds = %123, %116
  br label %125

125:                                              ; preds = %124
  br label %126

126:                                              ; preds = %125
  br i1 false, label %127, label %129

127:                                              ; preds = %126
  br i1 false, label %131, label %128

128:                                              ; preds = %127
  br label %132

129:                                              ; preds = %126
  br label %132

130:                                              ; preds = %120
  unreachable

131:                                              ; preds = %127
  unreachable

132:                                              ; preds = %129, %128
  br label %134

133:                                              ; preds = %114
  br label %209

134:                                              ; preds = %132, %101, %88
  br label %84

135:                                              ; preds = %77
  br label %139

136:                                              ; preds = %49
  br label %137

137:                                              ; preds = %136
  br label %138

138:                                              ; preds = %137
  br label %140

139:                                              ; preds = %135
  br label %159

140:                                              ; preds = %138
  br i1 false, label %141, label %142

141:                                              ; preds = %140
  unreachable

142:                                              ; preds = %140
  br i1 false, label %143, label %144

143:                                              ; preds = %142
  unreachable

144:                                              ; preds = %142
  br label %145

145:                                              ; preds = %144
  br i1 false, label %146, label %150

146:                                              ; preds = %145
  br i1 false, label %147, label %148

147:                                              ; preds = %146
  unreachable

148:                                              ; preds = %146
  br label %149

149:                                              ; preds = %148
  br i1 false, label %151, label %157

150:                                              ; preds = %145
  br label %152

151:                                              ; preds = %149
  br label %154

152:                                              ; preds = %150
  br label %153

153:                                              ; preds = %152
  unreachable

154:                                              ; preds = %151
  br label %155

155:                                              ; preds = %154
  br label %156

156:                                              ; preds = %155
  unreachable

157:                                              ; preds = %149
  br label %158

158:                                              ; preds = %157
  br label %159

159:                                              ; preds = %158, %139
  br i1 false, label %172, label %160

160:                                              ; preds = %159
  br label %161

161:                                              ; preds = %160
  br i1 false, label %162, label %163

162:                                              ; preds = %161
  br label %171

163:                                              ; preds = %161
  br label %164

164:                                              ; preds = %163
  br label %165

165:                                              ; preds = %165, %164
  br i1 false, label %165, label %166

166:                                              ; preds = %165
  br label %167

167:                                              ; preds = %166
  br label %168

168:                                              ; preds = %167
  br i1 false, label %169, label %170

169:                                              ; preds = %168
  unreachable

170:                                              ; preds = %168
  br label %171

171:                                              ; preds = %170, %162
  br label %194

172:                                              ; preds = %159
  br i1 false, label %191, label %173

173:                                              ; preds = %172
  br label %174

174:                                              ; preds = %173
  br i1 false, label %175, label %182

175:                                              ; preds = %174
  br label %176

176:                                              ; preds = %175
  br i1 false, label %178, label %177

177:                                              ; preds = %176
  unreachable

178:                                              ; preds = %176
  br i1 false, label %188, label %179

179:                                              ; preds = %178
  br i1 false, label %216, label %180

180:                                              ; preds = %179
  br label %181

181:                                              ; preds = %180
  br label %182

182:                                              ; preds = %181, %174
  br label %183

183:                                              ; preds = %182
  br label %184

184:                                              ; preds = %183
  br i1 false, label %185, label %187

185:                                              ; preds = %184
  br i1 false, label %189, label %186

186:                                              ; preds = %185
  br label %190

187:                                              ; preds = %184
  br label %190

188:                                              ; preds = %178
  unreachable

189:                                              ; preds = %185
  unreachable

190:                                              ; preds = %187, %186
  br label %194

191:                                              ; preds = %172
  br i1 false, label %209, label %192

192:                                              ; preds = %191
  br label %193

193:                                              ; preds = %192
  br label %194

194:                                              ; preds = %193, %190, %171
  br label %195

195:                                              ; preds = %194
  br i1 false, label %196, label %200

196:                                              ; preds = %195
  br label %7

197:                                              ; preds = %204
  br i1 false, label %198, label %199

198:                                              ; preds = %197
  unreachable

199:                                              ; preds = %197
  br label %205

200:                                              ; preds = %195
  switch i8 0, label %202 [
    i8 46, label %204
    i8 57, label %201
  ]

201:                                              ; preds = %200
  br label %204

202:                                              ; preds = %200
  br label %203

203:                                              ; preds = %202
  br label %205

204:                                              ; preds = %201, %200
  br label %197

205:                                              ; preds = %203, %199
  br i1 false, label %206, label %208

206:                                              ; preds = %205
  br label %207

207:                                              ; preds = %206
  unreachable

208:                                              ; preds = %205
  br label %209

209:                                              ; preds = %208, %191, %133, %90, %73, %25, %10
  ret i8 0

210:                                              ; preds = %30
  unreachable

211:                                              ; preds = %57
  unreachable

212:                                              ; preds = %78
  unreachable

213:                                              ; No predecessors!
  unreachable

214:                                              ; preds = %95
  unreachable

215:                                              ; preds = %121
  unreachable

216:                                              ; preds = %179
  unreachable
}
