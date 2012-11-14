target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"
; RUN: opt < %s -basicaa -mtriple=x86_64-unknown-linux-gnu -mcpu=corei7 -bb-vectorize -S | FileCheck %s

%struct.gsm_state.2.8.14.15.16.17.19.22.23.25.26.28.29.31.32.33.35.36.37.38.40.41.42.44.45.47.48.50.52.53.54.56.57.58.59.60.61.62.63.66.73.83.84.89.90.91.92.93.94.95.96.99.100.101.102.103.104.106.107.114.116.121.122.129.130.135.136.137.138.139.140.141.142.143.144.147.148.149.158.159.160.161.164.165.166.167.168.169.172.179.181.182.183.188.195.200.201.202.203.204.205.208.209.210.212.213.214.215.222.223.225.226.230.231.232.233.234.235.236.237.238.239.240.241.242.243.244.352 = type { [280 x i16], i16, i64, i32, [8 x i16], [2 x [8 x i16]], i16, i16, [9 x i16], i16, i8, i8 }

define void @gsm_encode(%struct.gsm_state.2.8.14.15.16.17.19.22.23.25.26.28.29.31.32.33.35.36.37.38.40.41.42.44.45.47.48.50.52.53.54.56.57.58.59.60.61.62.63.66.73.83.84.89.90.91.92.93.94.95.96.99.100.101.102.103.104.106.107.114.116.121.122.129.130.135.136.137.138.139.140.141.142.143.144.147.148.149.158.159.160.161.164.165.166.167.168.169.172.179.181.182.183.188.195.200.201.202.203.204.205.208.209.210.212.213.214.215.222.223.225.226.230.231.232.233.234.235.236.237.238.239.240.241.242.243.244.352* %s, i16* %source, i8* %c) nounwind uwtable {
entry:
  %xmc = alloca [52 x i16], align 16
  %arraydecay5 = getelementptr inbounds [52 x i16]* %xmc, i64 0, i64 0
  call void @Gsm_Coder(%struct.gsm_state.2.8.14.15.16.17.19.22.23.25.26.28.29.31.32.33.35.36.37.38.40.41.42.44.45.47.48.50.52.53.54.56.57.58.59.60.61.62.63.66.73.83.84.89.90.91.92.93.94.95.96.99.100.101.102.103.104.106.107.114.116.121.122.129.130.135.136.137.138.139.140.141.142.143.144.147.148.149.158.159.160.161.164.165.166.167.168.169.172.179.181.182.183.188.195.200.201.202.203.204.205.208.209.210.212.213.214.215.222.223.225.226.230.231.232.233.234.235.236.237.238.239.240.241.242.243.244.352* %s, i16* %source, i16* undef, i16* null, i16* undef, i16* undef, i16* undef, i16* %arraydecay5) nounwind
  %incdec.ptr136 = getelementptr inbounds i8* %c, i64 10
  %incdec.ptr157 = getelementptr inbounds i8* %c, i64 11
  store i8 0, i8* %incdec.ptr136, align 1
  %arrayidx162 = getelementptr inbounds [52 x i16]* %xmc, i64 0, i64 11
  %0 = load i16* %arrayidx162, align 2
  %conv1631 = trunc i16 %0 to i8
  %and164 = shl i8 %conv1631, 3
  %shl165 = and i8 %and164, 56
  %incdec.ptr172 = getelementptr inbounds i8* %c, i64 12
  store i8 %shl165, i8* %incdec.ptr157, align 1
  %1 = load i16* inttoptr (i64 2 to i16*), align 2
  %conv1742 = trunc i16 %1 to i8
  %and175 = shl i8 %conv1742, 1
  %incdec.ptr183 = getelementptr inbounds i8* %c, i64 13
  store i8 %and175, i8* %incdec.ptr172, align 1
  %incdec.ptr199 = getelementptr inbounds i8* %c, i64 14
  store i8 0, i8* %incdec.ptr183, align 1
  %arrayidx214 = getelementptr inbounds [52 x i16]* %xmc, i64 0, i64 15
  %incdec.ptr220 = getelementptr inbounds i8* %c, i64 15
  store i8 0, i8* %incdec.ptr199, align 1
  %2 = load i16* %arrayidx214, align 2
  %conv2223 = trunc i16 %2 to i8
  %and223 = shl i8 %conv2223, 6
  %incdec.ptr235 = getelementptr inbounds i8* %c, i64 16
  store i8 %and223, i8* %incdec.ptr220, align 1
  %arrayidx240 = getelementptr inbounds [52 x i16]* %xmc, i64 0, i64 19
  %3 = load i16* %arrayidx240, align 2
  %conv2414 = trunc i16 %3 to i8
  %and242 = shl i8 %conv2414, 2
  %shl243 = and i8 %and242, 28
  %incdec.ptr251 = getelementptr inbounds i8* %c, i64 17
  store i8 %shl243, i8* %incdec.ptr235, align 1
  %incdec.ptr272 = getelementptr inbounds i8* %c, i64 18
  store i8 0, i8* %incdec.ptr251, align 1
  %arrayidx282 = getelementptr inbounds [52 x i16]* %xmc, i64 0, i64 25
  %4 = load i16* %arrayidx282, align 2
  %conv2835 = trunc i16 %4 to i8
  %and284 = and i8 %conv2835, 7
  %incdec.ptr287 = getelementptr inbounds i8* %c, i64 19
  store i8 %and284, i8* %incdec.ptr272, align 1
  %incdec.ptr298 = getelementptr inbounds i8* %c, i64 20
  store i8 0, i8* %incdec.ptr287, align 1
  %incdec.ptr314 = getelementptr inbounds i8* %c, i64 21
  store i8 0, i8* %incdec.ptr298, align 1
  %arrayidx319 = getelementptr inbounds [52 x i16]* %xmc, i64 0, i64 26
  %5 = load i16* %arrayidx319, align 4
  %conv3206 = trunc i16 %5 to i8
  %and321 = shl i8 %conv3206, 4
  %shl322 = and i8 %and321, 112
  %incdec.ptr335 = getelementptr inbounds i8* %c, i64 22
  store i8 %shl322, i8* %incdec.ptr314, align 1
  %arrayidx340 = getelementptr inbounds [52 x i16]* %xmc, i64 0, i64 29
  %6 = load i16* %arrayidx340, align 2
  %conv3417 = trunc i16 %6 to i8
  %and342 = shl i8 %conv3417, 3
  %shl343 = and i8 %and342, 56
  %incdec.ptr350 = getelementptr inbounds i8* %c, i64 23
  store i8 %shl343, i8* %incdec.ptr335, align 1
  %incdec.ptr366 = getelementptr inbounds i8* %c, i64 24
  store i8 0, i8* %incdec.ptr350, align 1
  %arrayidx381 = getelementptr inbounds [52 x i16]* %xmc, i64 0, i64 36
  %incdec.ptr387 = getelementptr inbounds i8* %c, i64 25
  store i8 0, i8* %incdec.ptr366, align 1
  %7 = load i16* %arrayidx381, align 8
  %conv3898 = trunc i16 %7 to i8
  %and390 = shl i8 %conv3898, 6
  store i8 %and390, i8* %incdec.ptr387, align 1
  unreachable
; CHECK: @gsm_encode
}

declare void @Gsm_Coder(%struct.gsm_state.2.8.14.15.16.17.19.22.23.25.26.28.29.31.32.33.35.36.37.38.40.41.42.44.45.47.48.50.52.53.54.56.57.58.59.60.61.62.63.66.73.83.84.89.90.91.92.93.94.95.96.99.100.101.102.103.104.106.107.114.116.121.122.129.130.135.136.137.138.139.140.141.142.143.144.147.148.149.158.159.160.161.164.165.166.167.168.169.172.179.181.182.183.188.195.200.201.202.203.204.205.208.209.210.212.213.214.215.222.223.225.226.230.231.232.233.234.235.236.237.238.239.240.241.242.243.244.352*, i16*, i16*, i16*, i16*, i16*, i16*, i16*)

declare void @llvm.trap() noreturn nounwind
