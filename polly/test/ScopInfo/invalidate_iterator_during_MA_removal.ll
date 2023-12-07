; RUN: opt %loadPolly -polly-print-scops -disable-output < %s
;
; Check that no invalidated iterator is accessed while elements from
; the list of MemoryAccesses are removed.
; No CHECK-line because the with no MemoryAccesses left,
; we cannot model a SCoP.
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

%struct.vorbis_dsp_state.29.212.395.700.761.944.1066.1127.1188.2825.2980.1.51.76.101.126.780 = type { i32, ptr, ptr, ptr, i32, i32, i32, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, ptr }
%struct.vorbis_info.28.211.394.699.760.943.1065.1126.1187.2824.2979.0.50.75.100.125.779 = type { i32, i32, i64, i64, i64, i64, i64, ptr }
%struct.codec_setup_info.59.242.425.730.791.974.1096.1157.1218.2855.2992.13.63.88.113.138.792 = type { [2 x i64], i32, i32, i32, i32, i32, i32, [64 x ptr], [64 x i32], [64 x ptr], [64 x i32], [64 x ptr], [64 x i32], [64 x ptr], [256 x ptr], ptr, [4 x ptr], %struct.vorbis_info_psy_global.13.196.379.684.745.928.1050.1111.1172.2809.2988.9.59.84.109.134.788, %struct.bitrate_manager_info.56.239.422.727.788.971.1093.1154.1215.2852.2989.10.60.85.110.135.789, %struct.highlevel_encode_setup.58.241.424.729.790.973.1095.1156.1217.2854.2991.12.62.87.112.137.791, i32 }
%struct.vorbis_info_mode.36.219.402.707.768.951.1073.1134.1195.2832.2981.2.52.77.102.127.781 = type { i32, i32, i32, i32 }
%struct.static_codebook.18.201.384.689.750.933.1055.1116.1177.2814.2985.6.56.81.106.131.785 = type { i64, i64, ptr, i32, i64, i64, i32, i32, ptr, ptr, ptr, ptr, i32 }
%struct.encode_aux_nearestmatch.15.198.381.686.747.930.1052.1113.1174.2811.2982.3.53.78.103.128.782 = type { ptr, ptr, ptr, ptr, i64, i64 }
%struct.encode_aux_threshmatch.16.199.382.687.748.931.1053.1114.1175.2812.2983.4.54.79.104.129.783 = type { ptr, ptr, i32, i32 }
%struct.encode_aux_pigeonhole.17.200.383.688.749.932.1054.1115.1176.2813.2984.5.55.80.105.130.784 = type { float, float, i32, i32, ptr, i64, ptr, ptr, ptr }
%struct.codebook.43.226.409.714.775.958.1080.1141.1202.2839.2986.7.57.82.107.132.786 = type { i64, i64, i64, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32 }
%struct.vorbis_info_psy.35.218.401.706.767.950.1072.1133.1194.2831.2987.8.58.83.108.133.787 = type { i32, float, float, [3 x float], float, float, float, [17 x float], i32, float, float, float, i32, i32, i32, [3 x [17 x float]], [40 x float], float, i32, i32, i32, i32, double }
%struct.vorbis_info_psy_global.13.196.379.684.745.928.1050.1111.1172.2809.2988.9.59.84.109.134.788 = type { i32, [7 x float], [7 x float], float, float, float, [15 x i32], [2 x [15 x i32]], [15 x i32], [15 x i32], [2 x [15 x i32]] }
%struct.bitrate_manager_info.56.239.422.727.788.971.1093.1154.1215.2852.2989.10.60.85.110.135.789 = type { double, double, double, double, double, double, double, double, double }
%struct.highlevel_encode_setup.58.241.424.729.790.973.1095.1156.1217.2854.2991.12.62.87.112.137.791 = type { ptr, i32, double, double, double, double, i32, i64, i64, i64, i64, double, double, double, i32, i32, double, double, double, double, double, double, [4 x %struct.highlevel_byblocktype.57.240.423.728.789.972.1094.1155.1216.2853.2990.11.61.86.111.136.790] }
%struct.highlevel_byblocktype.57.240.423.728.789.972.1094.1155.1216.2853.2990.11.61.86.111.136.790 = type { double, double, double, double }
%struct.private_state.60.243.426.731.792.975.1097.1158.1219.2856.3003.24.74.99.124.149.803 = type { ptr, [2 x i32], [2 x ptr], [2 x %struct.drft_lookup.51.234.417.722.783.966.1088.1149.1210.2847.2997.18.68.93.118.143.797], i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, %struct.bitrate_manager_state.49.232.415.720.781.964.1086.1147.1208.2845.3002.23.73.98.123.148.802, i64 }
%struct.envelope_lookup.48.231.414.719.780.963.1085.1146.1207.2844.2996.17.67.92.117.142.796 = type { i32, i32, i32, float, %struct.mdct_lookup.45.228.411.716.777.960.1082.1143.1204.2841.2993.14.64.89.114.139.793, ptr, [7 x %struct.envelope_band.46.229.412.717.778.961.1083.1144.1205.2842.2994.15.65.90.115.140.794], ptr, i32, ptr, i64, i64, i64, i64 }
%struct.mdct_lookup.45.228.411.716.777.960.1082.1143.1204.2841.2993.14.64.89.114.139.793 = type { i32, i32, ptr, ptr, float }
%struct.envelope_band.46.229.412.717.778.961.1083.1144.1205.2842.2994.15.65.90.115.140.794 = type { i32, i32, ptr, float }
%struct.envelope_filter_state.47.230.413.718.779.962.1084.1145.1206.2843.2995.16.66.91.116.141.795 = type { [17 x float], i32, [15 x float], float, float, i32 }
%struct.drft_lookup.51.234.417.722.783.966.1088.1149.1210.2847.2997.18.68.93.118.143.797 = type { i32, ptr, ptr }
%struct.vorbis_look_psy.50.233.416.721.782.965.1087.1148.1209.2846.2998.19.69.94.119.144.798 = type { i32, ptr, ptr, ptr, ptr, ptr, ptr, i64, i64, i32, i32, i64 }
%struct.vorbis_look_psy_global.44.227.410.715.776.959.1081.1142.1203.2840.2999.20.70.95.120.145.799 = type { float, i32, ptr, [2 x [3 x i32]] }
%struct.bitrate_manager_state.49.232.415.720.781.964.1086.1147.1208.2845.3002.23.73.98.123.148.802 = type { ptr, ptr, i32, i32, i32, ptr, i32, i32, i32, i32, i32, i32, ptr, ptr, ptr, i64, i64, i32, i32, i32, i32, i32, double, ptr, ptr }
%struct.oggpack_buffer.27.210.393.698.759.942.1064.1125.1186.2823.3000.21.71.96.121.146.800 = type { i64, i32, ptr, ptr, i64 }
%struct.ogg_packet.39.222.405.710.771.954.1076.1137.1198.2835.3001.22.72.97.122.147.801 = type { ptr, i64, i64, i64, i64, i64 }

define void @vorbis_synthesis_blockin(ptr nocapture %v, ptr %A) {
entry:
  br label %entry.split

entry.split:                                      ; preds = %entry
  %vi1 = getelementptr inbounds %struct.vorbis_dsp_state.29.212.395.700.761.944.1066.1127.1188.2825.2980.1.51.76.101.126.780, ptr %v, i64 0, i32 1
  %tmp = load ptr, ptr %vi1, align 8
  %codec_setup = getelementptr inbounds %struct.vorbis_info.28.211.394.699.760.943.1065.1126.1187.2824.2979.0.50.75.100.125.779, ptr %tmp, i64 0, i32 7
  %tmp2 = load ptr, ptr %codec_setup, align 8
  %backend_state = getelementptr inbounds %struct.vorbis_dsp_state.29.212.395.700.761.944.1066.1127.1188.2825.2980.1.51.76.101.126.780, ptr %v, i64 0, i32 19
  %tmp4 = load ptr, ptr %backend_state, align 8
  br i1 false, label %cleanup, label %if.end

if.end:                                           ; preds = %entry.split
  %sample_count = getelementptr inbounds %struct.private_state.60.243.426.731.792.975.1097.1158.1219.2856.3003.24.74.99.124.149.803, ptr %tmp4, i64 0, i32 13
  store i64 -1, ptr %sample_count, align 8
  br label %cleanup

cleanup:                                          ; preds = %if.end, %entry.split
  ret void
}
