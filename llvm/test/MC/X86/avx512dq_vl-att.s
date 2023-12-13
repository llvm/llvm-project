# RUN: llvm-mc -triple x86_64 -show-encoding %s | FileCheck %s

# CHECK: vcvtps2qq	128(%rcx), %xmm2 {%k2} {z}
# CHECK: encoding: [0x62,0xf1,0x7d,0x8a,0x7b,0x51,0x10]
         vcvtps2qq	128(%rcx), %xmm2 {%k2} {z}
# CHECK: vcvtps2qq	128(%rcx), %xmm2 {%k2}
# CHECK: encoding: [0x62,0xf1,0x7d,0x0a,0x7b,0x51,0x10]
         vcvtps2qq	128(%rcx), %xmm2 {%k2}
# CHECK: vcvtps2qq	128(%rcx), %xmm2
# CHECK: encoding: [0x62,0xf1,0x7d,0x08,0x7b,0x51,0x10]
         vcvtps2qq	128(%rcx), %xmm2
# CHECK: vcvttps2qq	128(%rcx), %xmm1 {%k2} {z}
# CHECK: encoding: [0x62,0xf1,0x7d,0x8a,0x7a,0x49,0x10]
         vcvttps2qq	128(%rcx), %xmm1 {%k2} {z}
# CHECK: vcvttps2qq	128(%rcx), %xmm1 {%k2}
# CHECK: encoding: [0x62,0xf1,0x7d,0x0a,0x7a,0x49,0x10]
         vcvttps2qq	128(%rcx), %xmm1 {%k2}
# CHECK: vcvttps2qq	128(%rcx), %xmm1
# CHECK: encoding: [0x62,0xf1,0x7d,0x08,0x7a,0x49,0x10]
         vcvttps2qq	128(%rcx), %xmm1
# CHECK: vcvtps2uqq	128(%rcx), %xmm1 {%k2} {z}
# CHECK: encoding: [0x62,0xf1,0x7d,0x8a,0x79,0x49,0x10]
         vcvtps2uqq	128(%rcx), %xmm1 {%k2} {z}
# CHECK: vcvtps2uqq	128(%rcx), %xmm1 {%k2}
# CHECK: encoding: [0x62,0xf1,0x7d,0x0a,0x79,0x49,0x10]
         vcvtps2uqq	128(%rcx), %xmm1 {%k2}
# CHECK: vcvtps2uqq	128(%rcx), %xmm1
# CHECK: encoding: [0x62,0xf1,0x7d,0x08,0x79,0x49,0x10]
         vcvtps2uqq	128(%rcx), %xmm1
# CHECK: vcvttps2uqq	128(%rcx), %xmm1 {%k2} {z}
# CHECK: encoding: [0x62,0xf1,0x7d,0x8a,0x78,0x49,0x10]
         vcvttps2uqq	128(%rcx), %xmm1 {%k2} {z}
# CHECK: vcvttps2uqq	128(%rcx), %xmm1 {%k2}
# CHECK: encoding: [0x62,0xf1,0x7d,0x0a,0x78,0x49,0x10]
         vcvttps2uqq	128(%rcx), %xmm1 {%k2}
# CHECK: vcvttps2uqq	128(%rcx), %xmm1
# CHECK: encoding: [0x62,0xf1,0x7d,0x08,0x78,0x49,0x10]
         vcvttps2uqq	128(%rcx), %xmm1
# CHECK: vcvtps2qq	128(%rcx), %xmm2 {%k2} {z}
# CHECK: encoding: [0x62,0xf1,0x7d,0x8a,0x7b,0x51,0x10]
         vcvtps2qq	128(%rcx), %xmm2 {%k2} {z}
# CHECK: vcvtps2qq	128(%rcx), %xmm2 {%k2}
# CHECK: encoding: [0x62,0xf1,0x7d,0x0a,0x7b,0x51,0x10]
         vcvtps2qq	128(%rcx), %xmm2 {%k2}
# CHECK: vcvtps2qq	128(%rcx), %xmm2
# CHECK: encoding: [0x62,0xf1,0x7d,0x08,0x7b,0x51,0x10]
         vcvtps2qq	128(%rcx), %xmm2
# CHECK: vcvttps2qq	128(%rcx), %xmm1 {%k2} {z}
# CHECK: encoding: [0x62,0xf1,0x7d,0x8a,0x7a,0x49,0x10]
         vcvttps2qq	128(%rcx), %xmm1 {%k2} {z}
# CHECK: vcvttps2qq	128(%rcx), %xmm1 {%k2}
# CHECK: encoding: [0x62,0xf1,0x7d,0x0a,0x7a,0x49,0x10]
         vcvttps2qq	128(%rcx), %xmm1 {%k2}
# CHECK: vcvttps2qq	128(%rcx), %xmm1
# CHECK: encoding: [0x62,0xf1,0x7d,0x08,0x7a,0x49,0x10]
         vcvttps2qq	128(%rcx), %xmm1
# CHECK: vcvtps2uqq	128(%rcx), %xmm1 {%k2} {z}
# CHECK: encoding: [0x62,0xf1,0x7d,0x8a,0x79,0x49,0x10]
         vcvtps2uqq	128(%rcx), %xmm1 {%k2} {z}
# CHECK: vcvtps2uqq	128(%rcx), %xmm1 {%k2}
# CHECK: encoding: [0x62,0xf1,0x7d,0x0a,0x79,0x49,0x10]
         vcvtps2uqq	128(%rcx), %xmm1 {%k2}
# CHECK: vcvtps2uqq	128(%rcx), %xmm1
# CHECK: encoding: [0x62,0xf1,0x7d,0x08,0x79,0x49,0x10]
         vcvtps2uqq	128(%rcx), %xmm1
# CHECK: vcvttps2uqq	128(%rcx), %xmm1 {%k2} {z}
# CHECK: encoding: [0x62,0xf1,0x7d,0x8a,0x78,0x49,0x10]
         vcvttps2uqq	128(%rcx), %xmm1 {%k2} {z}
# CHECK: vcvttps2uqq	128(%rcx), %xmm1 {%k2}
# CHECK: encoding: [0x62,0xf1,0x7d,0x0a,0x78,0x49,0x10]
         vcvttps2uqq	128(%rcx), %xmm1 {%k2}
# CHECK: vcvttps2uqq	128(%rcx), %xmm1
# CHECK: encoding: [0x62,0xf1,0x7d,0x08,0x78,0x49,0x10]
         vcvttps2uqq	128(%rcx), %xmm1
# CHECK: vfpclasspd	$171, %xmm18, %k2
# CHECK: encoding: [0x62,0xb3,0xfd,0x08,0x66,0xd2,0xab]
         vfpclasspd	$171, %xmm18, %k2
# CHECK: vfpclasspd	$171, %xmm18, %k2 {%k7}
# CHECK: encoding: [0x62,0xb3,0xfd,0x0f,0x66,0xd2,0xab]
         vfpclasspd	$171, %xmm18, %k2 {%k7}
# CHECK: vfpclasspdx	$123, (%rcx), %k2
# CHECK: encoding: [0x62,0xf3,0xfd,0x08,0x66,0x11,0x7b]
         vfpclasspdx	$123, (%rcx), %k2
# CHECK: vfpclasspdx	$123, (%rcx), %k2 {%k7}
# CHECK: encoding: [0x62,0xf3,0xfd,0x0f,0x66,0x11,0x7b]
         vfpclasspdx	$123, (%rcx), %k2 {%k7}
# CHECK: vfpclasspd	$123, (%rcx){1to2}, %k2
# CHECK: encoding: [0x62,0xf3,0xfd,0x18,0x66,0x11,0x7b]
         vfpclasspd	$123, (%rcx){1to2}, %k2
# CHECK: vfpclasspd	$123, (%rcx){1to2}, %k2 {%k7}
# CHECK: encoding: [0x62,0xf3,0xfd,0x1f,0x66,0x11,0x7b]
         vfpclasspd	$123, (%rcx){1to2}, %k2 {%k7}
# CHECK: vfpclassps	$171, %xmm18, %k2
# CHECK: encoding: [0x62,0xb3,0x7d,0x08,0x66,0xd2,0xab]
         vfpclassps	$171, %xmm18, %k2
# CHECK: vfpclassps	$171, %xmm18, %k2 {%k7}
# CHECK: encoding: [0x62,0xb3,0x7d,0x0f,0x66,0xd2,0xab]
         vfpclassps	$171, %xmm18, %k2 {%k7}
# CHECK: vfpclasspsx	$123, (%rcx), %k2
# CHECK: encoding: [0x62,0xf3,0x7d,0x08,0x66,0x11,0x7b]
         vfpclasspsx	$123, (%rcx), %k2
# CHECK: vfpclasspsx	$123, (%rcx), %k2 {%k7}
# CHECK: encoding: [0x62,0xf3,0x7d,0x0f,0x66,0x11,0x7b]
         vfpclasspsx	$123, (%rcx), %k2 {%k7}
# CHECK: vfpclassps	$123, (%rcx){1to4}, %k2
# CHECK: encoding: [0x62,0xf3,0x7d,0x18,0x66,0x11,0x7b]
         vfpclassps	$123, (%rcx){1to4}, %k2
# CHECK: vfpclassps	$123, (%rcx){1to4}, %k2 {%k7}
# CHECK: encoding: [0x62,0xf3,0x7d,0x1f,0x66,0x11,0x7b]
         vfpclassps	$123, (%rcx){1to4}, %k2 {%k7}
# CHECK: vfpclasspd	$171, %ymm18, %k2
# CHECK: encoding: [0x62,0xb3,0xfd,0x28,0x66,0xd2,0xab]
         vfpclasspd	$171, %ymm18, %k2
# CHECK: vfpclasspd	$171, %ymm18, %k2 {%k7}
# CHECK: encoding: [0x62,0xb3,0xfd,0x2f,0x66,0xd2,0xab]
         vfpclasspd	$171, %ymm18, %k2 {%k7}
# CHECK: vfpclasspdy	$123, (%rcx), %k2
# CHECK: encoding: [0x62,0xf3,0xfd,0x28,0x66,0x11,0x7b]
         vfpclasspdy	$123, (%rcx), %k2
# CHECK: vfpclasspdy	$123, (%rcx), %k2 {%k7}
# CHECK: encoding: [0x62,0xf3,0xfd,0x2f,0x66,0x11,0x7b]
         vfpclasspdy	$123, (%rcx), %k2 {%k7}
# CHECK: vfpclasspd	$123, (%rcx){1to4}, %k2
# CHECK: encoding: [0x62,0xf3,0xfd,0x38,0x66,0x11,0x7b]
         vfpclasspd	$123, (%rcx){1to4}, %k2
# CHECK: vfpclasspd	$123, (%rcx){1to4}, %k2 {%k7}
# CHECK: encoding: [0x62,0xf3,0xfd,0x3f,0x66,0x11,0x7b]
         vfpclasspd	$123, (%rcx){1to4}, %k2 {%k7}
# CHECK: vfpclassps	$171, %ymm18, %k2
# CHECK: encoding: [0x62,0xb3,0x7d,0x28,0x66,0xd2,0xab]
         vfpclassps	$171, %ymm18, %k2
# CHECK: vfpclassps	$171, %ymm18, %k2 {%k7}
# CHECK: encoding: [0x62,0xb3,0x7d,0x2f,0x66,0xd2,0xab]
         vfpclassps	$171, %ymm18, %k2 {%k7}
# CHECK: vfpclasspsy	$123, (%rcx), %k2
# CHECK: encoding: [0x62,0xf3,0x7d,0x28,0x66,0x11,0x7b]
         vfpclasspsy	$123, (%rcx), %k2
# CHECK: vfpclasspsy	$123, (%rcx), %k2 {%k7}
# CHECK: encoding: [0x62,0xf3,0x7d,0x2f,0x66,0x11,0x7b]
         vfpclasspsy	$123, (%rcx), %k2 {%k7}
# CHECK: vfpclassps	$123, (%rcx){1to8}, %k2
# CHECK: encoding: [0x62,0xf3,0x7d,0x38,0x66,0x11,0x7b]
         vfpclassps	$123, (%rcx){1to8}, %k2
# CHECK: vfpclassps	$123, (%rcx){1to8}, %k2 {%k7}
# CHECK: encoding: [0x62,0xf3,0x7d,0x3f,0x66,0x11,0x7b]
         vfpclassps	$123, (%rcx){1to8}, %k2 {%k7}
# CHECK: vcvttps2uqq     128(%ecx), %xmm1 {%k2}
# CHECK: encoding: [0x67,0x62,0xf1,0x7d,0x0a,0x78,0x49,0x10]
         vcvttps2uqq     128(%ecx), %xmm1 {%k2}
