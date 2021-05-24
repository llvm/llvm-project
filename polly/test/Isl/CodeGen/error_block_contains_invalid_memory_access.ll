; RUN: opt %loadPolly -polly-codegen < %s
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

%struct.img_par.12.33.54.243.348.558.600.852.999.1209.2070.2154.2175.2238.2259.2280.2322 = type { i32, i32, i32, i32, i32*, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, [16 x [16 x i16]], [6 x [32 x i32]], [16 x [16 x i32]], [4 x [12 x [4 x [4 x i32]]]], [16 x i32], i8**, i32*, i32***, i32**, i32, i32, i32, i32, %struct.Slice.8.29.50.239.344.554.596.848.995.1205.2066.2150.2171.2234.2255.2276.2318*, %struct.macroblock.9.30.51.240.345.555.597.849.996.1206.2067.2151.2172.2235.2256.2277.2319*, i32, i32, i32, i32, i32, i32, %struct.DecRefPicMarking_s.10.31.52.241.346.556.598.850.997.1207.2068.2152.2173.2236.2257.2278.2320*, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, [3 x i32], i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32***, i32***, i32****, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, [3 x [2 x i32]], [3 x [2 x i32]], i32, i32, i64, i64, %struct.timeb.11.32.53.242.347.557.599.851.998.1208.2069.2153.2174.2237.2258.2279.2321, %struct.timeb.11.32.53.242.347.557.599.851.998.1208.2069.2153.2174.2237.2258.2279.2321, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 }
%struct.Slice.8.29.50.239.344.554.596.848.995.1205.2066.2150.2171.2234.2255.2276.2318 = type { i32, i32, i32, i32, i32, i32, i32, i32, i32, %struct.datapartition.3.24.45.234.339.549.591.843.990.1200.2061.2145.2166.2229.2250.2271.2313*, %struct.MotionInfoContexts.5.26.47.236.341.551.593.845.992.1202.2063.2147.2168.2231.2252.2273.2315*, %struct.TextureInfoContexts.6.27.48.237.342.552.594.846.993.1203.2064.2148.2169.2232.2253.2274.2316*, i32, i32*, i32*, i32*, i32, i32*, i32*, i32*, i32 (%struct.img_par.12.33.54.243.348.558.600.852.999.1209.2070.2154.2175.2238.2259.2280.2322*, %struct.inp_par.7.28.49.238.343.553.595.847.994.1204.2065.2149.2170.2233.2254.2275.2317*)*, i32, i32, i32, i32 }
%struct.datapartition.3.24.45.234.339.549.591.843.990.1200.2061.2145.2166.2229.2250.2271.2313 = type { %struct.Bitstream.0.21.42.231.336.546.588.840.987.1197.2058.2142.2163.2226.2247.2268.2310*, %struct.DecodingEnvironment.1.22.43.232.337.547.589.841.988.1198.2059.2143.2164.2227.2248.2269.2311, i32 (%struct.syntaxelement.2.23.44.233.338.548.590.842.989.1199.2060.2144.2165.2228.2249.2270.2312*, %struct.img_par.12.33.54.243.348.558.600.852.999.1209.2070.2154.2175.2238.2259.2280.2322*, %struct.datapartition.3.24.45.234.339.549.591.843.990.1200.2061.2145.2166.2229.2250.2271.2313*)* }
%struct.Bitstream.0.21.42.231.336.546.588.840.987.1197.2058.2142.2163.2226.2247.2268.2310 = type { i32, i32, i32, i32, i8*, i32 }
%struct.DecodingEnvironment.1.22.43.232.337.547.589.841.988.1198.2059.2143.2164.2227.2248.2269.2311 = type { i32, i32, i32, i32, i32, i8*, i32* }
%struct.syntaxelement.2.23.44.233.338.548.590.842.989.1199.2060.2144.2165.2228.2249.2270.2312 = type { i32, i32, i32, i32, i32, i32, i32, i32, void (i32, i32, i32*, i32*)*, void (%struct.syntaxelement.2.23.44.233.338.548.590.842.989.1199.2060.2144.2165.2228.2249.2270.2312*, %struct.img_par.12.33.54.243.348.558.600.852.999.1209.2070.2154.2175.2238.2259.2280.2322*, %struct.DecodingEnvironment.1.22.43.232.337.547.589.841.988.1198.2059.2143.2164.2227.2248.2269.2311*)* }
%struct.MotionInfoContexts.5.26.47.236.341.551.593.845.992.1202.2063.2147.2168.2231.2252.2273.2315 = type { [4 x [11 x %struct.BiContextType.4.25.46.235.340.550.592.844.991.1201.2062.2146.2167.2230.2251.2272.2314]], [2 x [9 x %struct.BiContextType.4.25.46.235.340.550.592.844.991.1201.2062.2146.2167.2230.2251.2272.2314]], [2 x [10 x %struct.BiContextType.4.25.46.235.340.550.592.844.991.1201.2062.2146.2167.2230.2251.2272.2314]], [2 x [6 x %struct.BiContextType.4.25.46.235.340.550.592.844.991.1201.2062.2146.2167.2230.2251.2272.2314]], [4 x %struct.BiContextType.4.25.46.235.340.550.592.844.991.1201.2062.2146.2167.2230.2251.2272.2314], [4 x %struct.BiContextType.4.25.46.235.340.550.592.844.991.1201.2062.2146.2167.2230.2251.2272.2314], [3 x %struct.BiContextType.4.25.46.235.340.550.592.844.991.1201.2062.2146.2167.2230.2251.2272.2314] }
%struct.BiContextType.4.25.46.235.340.550.592.844.991.1201.2062.2146.2167.2230.2251.2272.2314 = type { i16, i8 }
%struct.TextureInfoContexts.6.27.48.237.342.552.594.846.993.1203.2064.2148.2169.2232.2253.2274.2316 = type { [2 x %struct.BiContextType.4.25.46.235.340.550.592.844.991.1201.2062.2146.2167.2230.2251.2272.2314], [4 x %struct.BiContextType.4.25.46.235.340.550.592.844.991.1201.2062.2146.2167.2230.2251.2272.2314], [3 x [4 x %struct.BiContextType.4.25.46.235.340.550.592.844.991.1201.2062.2146.2167.2230.2251.2272.2314]], [10 x [4 x %struct.BiContextType.4.25.46.235.340.550.592.844.991.1201.2062.2146.2167.2230.2251.2272.2314]], [10 x [15 x %struct.BiContextType.4.25.46.235.340.550.592.844.991.1201.2062.2146.2167.2230.2251.2272.2314]], [10 x [15 x %struct.BiContextType.4.25.46.235.340.550.592.844.991.1201.2062.2146.2167.2230.2251.2272.2314]], [10 x [5 x %struct.BiContextType.4.25.46.235.340.550.592.844.991.1201.2062.2146.2167.2230.2251.2272.2314]], [10 x [5 x %struct.BiContextType.4.25.46.235.340.550.592.844.991.1201.2062.2146.2167.2230.2251.2272.2314]], [10 x [15 x %struct.BiContextType.4.25.46.235.340.550.592.844.991.1201.2062.2146.2167.2230.2251.2272.2314]], [10 x [15 x %struct.BiContextType.4.25.46.235.340.550.592.844.991.1201.2062.2146.2167.2230.2251.2272.2314]] }
%struct.inp_par.7.28.49.238.343.553.595.847.994.1204.2065.2149.2170.2233.2254.2275.2317 = type { [1000 x i8], [1000 x i8], [1000 x i8], i32, i32, i32, i32, i32, i32, i32, i32 }
%struct.macroblock.9.30.51.240.345.555.597.849.996.1206.2067.2151.2172.2235.2256.2277.2319 = type { i32, [2 x i32], i32, i32, %struct.macroblock.9.30.51.240.345.555.597.849.996.1206.2067.2151.2172.2235.2256.2277.2319*, %struct.macroblock.9.30.51.240.345.555.597.849.996.1206.2067.2151.2172.2235.2256.2277.2319*, i32, [2 x [4 x [4 x [2 x i32]]]], i32, i64, i64, i32, i32, [4 x i8], [4 x i8], i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 }
%struct.DecRefPicMarking_s.10.31.52.241.346.556.598.850.997.1207.2068.2152.2173.2236.2257.2278.2320 = type { i32, i32, i32, i32, i32, %struct.DecRefPicMarking_s.10.31.52.241.346.556.598.850.997.1207.2068.2152.2173.2236.2257.2278.2320* }
%struct.timeb.11.32.53.242.347.557.599.851.998.1208.2069.2153.2174.2237.2258.2279.2321 = type { i64, i16, i16, i16 }
%struct.pix_pos.13.34.55.244.349.559.601.853.1000.1210.2071.2155.2176.2239.2260.2281.2323 = type { i32, i32, i32, i32, i32, i32 }

declare void @getLuma4x4Neighbour() #0

; Function Attrs: nounwind uwtable
define void @readCBP_CABAC(%struct.img_par.12.33.54.243.348.558.600.852.999.1209.2070.2154.2175.2238.2259.2280.2322* %img) #1 {
entry:
  %block_a = alloca %struct.pix_pos.13.34.55.244.349.559.601.853.1000.1210.2071.2155.2176.2239.2260.2281.2323, align 4
  %mb_data = getelementptr inbounds %struct.img_par.12.33.54.243.348.558.600.852.999.1209.2070.2154.2175.2238.2259.2280.2322, %struct.img_par.12.33.54.243.348.558.600.852.999.1209.2070.2154.2175.2238.2259.2280.2322* %img, i64 0, i32 39
  br label %for.cond.1.preheader

for.cond.1.preheader:                             ; preds = %for.inc.84, %entry
  br label %for.body.3

for.body.3:                                       ; preds = %if.end.72, %for.cond.1.preheader
  %mb_x.056 = phi i32 [ 0, %for.cond.1.preheader ], [ undef, %if.end.72 ]
  br i1 undef, label %if.end.35, label %if.else.14

if.else.14:                                       ; preds = %for.body.3
  br i1 undef, label %if.end.35, label %if.else.19

if.else.19:                                       ; preds = %if.else.14
  br label %if.end.35

if.end.35:                                        ; preds = %if.else.19, %if.else.14, %for.body.3
  %b.0 = zext i1 undef to i64
  %cmp36 = icmp eq i32 %mb_x.056, 0
  br i1 %cmp36, label %if.then.38, label %if.else.66

if.then.38:                                       ; preds = %if.end.35
  call void @getLuma4x4Neighbour() #2
  %0 = load i32, i32* null, align 4
  %tobool = icmp eq i32 %0, 0
  br i1 %tobool, label %if.end.72, label %if.then.42

if.then.42:                                       ; preds = %if.then.38
  %mb_addr = getelementptr inbounds %struct.pix_pos.13.34.55.244.349.559.601.853.1000.1210.2071.2155.2176.2239.2260.2281.2323, %struct.pix_pos.13.34.55.244.349.559.601.853.1000.1210.2071.2155.2176.2239.2260.2281.2323* %block_a, i64 0, i32 1
  %1 = load %struct.macroblock.9.30.51.240.345.555.597.849.996.1206.2067.2151.2172.2235.2256.2277.2319*, %struct.macroblock.9.30.51.240.345.555.597.849.996.1206.2067.2151.2172.2235.2256.2277.2319** %mb_data, align 8
  %mb_type46 = getelementptr inbounds %struct.macroblock.9.30.51.240.345.555.597.849.996.1206.2067.2151.2172.2235.2256.2277.2319, %struct.macroblock.9.30.51.240.345.555.597.849.996.1206.2067.2151.2172.2235.2256.2277.2319* %1, i64 0, i32 6
  br i1 false, label %if.end.72, label %if.else.50

if.else.50:                                       ; preds = %if.then.42
  br label %if.end.72

if.else.66:                                       ; preds = %if.end.35
  br label %if.end.72

if.end.72:                                        ; preds = %if.else.66, %if.else.50, %if.then.42, %if.then.38
  %mul73 = shl nuw nsw i64 %b.0, 1
  br i1 undef, label %for.body.3, label %for.inc.84

for.inc.84:                                       ; preds = %if.end.72
  br i1 undef, label %for.cond.1.preheader, label %for.end.86

for.end.86:                                       ; preds = %for.inc.84
  ret void
}

attributes #0 = { "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="haswell" "target-features"="+aes,+avx,+avx2,+bmi,+bmi2,+cmov,+cx16,+f16c,+fma,+fsgsbase,+fxsr,+hle,+lzcnt,+mmx,+movbe,+pclmul,+popcnt,+rdrnd,+rtm,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+xsave,+xsaveopt,-adx,-avx512bw,-avx512cd,-avx512dq,-avx512er,-avx512f,-avx512pf,-avx512vl,-fma4,-prfchw,-rdseed,-sha,-sse4a,-tbm,-xop,-xsavec,-xsaves" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind uwtable "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="haswell" "target-features"="+aes,+avx,+avx2,+bmi,+bmi2,+cmov,+cx16,+f16c,+fma,+fsgsbase,+fxsr,+hle,+lzcnt,+mmx,+movbe,+pclmul,+popcnt,+rdrnd,+rtm,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+xsave,+xsaveopt,-adx,-avx512bw,-avx512cd,-avx512dq,-avx512er,-avx512f,-avx512pf,-avx512vl,-fma4,-prfchw,-rdseed,-sha,-sse4a,-tbm,-xop,-xsavec,-xsaves" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind }
