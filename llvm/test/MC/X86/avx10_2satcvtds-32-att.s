// RUN: llvm-mc -triple i386 --show-encoding %s | FileCheck %s

// CHECK: vcvttsd2sis %xmm2, %ecx
// CHECK: encoding: [0x62,0xf5,0x7f,0x08,0x6d,0xca]
          vcvttsd2sis %xmm2, %ecx

// CHECK: vcvttsd2sis {sae}, %xmm2, %ecx
// CHECK: encoding: [0x62,0xf5,0x7f,0x18,0x6d,0xca]
          vcvttsd2sis {sae}, %xmm2, %ecx

// CHECK: vcvttsd2sis  268435456(%esp,%esi,8), %ecx
// CHECK: encoding: [0x62,0xf5,0x7f,0x08,0x6d,0x8c,0xf4,0x00,0x00,0x00,0x10]
          vcvttsd2sis  268435456(%esp,%esi,8), %ecx

// CHECK: vcvttsd2sis  291(%edi,%eax,4), %ecx
// CHECK: encoding: [0x62,0xf5,0x7f,0x08,0x6d,0x8c,0x87,0x23,0x01,0x00,0x00]
          vcvttsd2sis  291(%edi,%eax,4), %ecx

// CHECK: vcvttsd2sis  (%eax), %ecx
// CHECK: encoding: [0x62,0xf5,0x7f,0x08,0x6d,0x08]
          vcvttsd2sis  (%eax), %ecx

// CHECK: vcvttsd2sis  -256(,%ebp,2), %ecx
// CHECK: encoding: [0x62,0xf5,0x7f,0x08,0x6d,0x0c,0x6d,0x00,0xff,0xff,0xff]
          vcvttsd2sis  -256(,%ebp,2), %ecx

// CHECK: vcvttsd2sis  1016(%ecx), %ecx
// CHECK: encoding: [0x62,0xf5,0x7f,0x08,0x6d,0x49,0x7f]
          vcvttsd2sis  1016(%ecx), %ecx

// CHECK: vcvttsd2sis  -1024(%edx), %ecx
// CHECK: encoding: [0x62,0xf5,0x7f,0x08,0x6d,0x4a,0x80]
          vcvttsd2sis  -1024(%edx), %ecx

// CHECK: vcvttsd2usis %xmm2, %ecx
// CHECK: encoding: [0x62,0xf5,0x7f,0x08,0x6c,0xca]
          vcvttsd2usis %xmm2, %ecx

// CHECK: vcvttsd2usis {sae}, %xmm2, %ecx
// CHECK: encoding: [0x62,0xf5,0x7f,0x18,0x6c,0xca]
          vcvttsd2usis {sae}, %xmm2, %ecx

// CHECK: vcvttsd2usis  268435456(%esp,%esi,8), %ecx
// CHECK: encoding: [0x62,0xf5,0x7f,0x08,0x6c,0x8c,0xf4,0x00,0x00,0x00,0x10]
          vcvttsd2usis  268435456(%esp,%esi,8), %ecx

// CHECK: vcvttsd2usis  291(%edi,%eax,4), %ecx
// CHECK: encoding: [0x62,0xf5,0x7f,0x08,0x6c,0x8c,0x87,0x23,0x01,0x00,0x00]
          vcvttsd2usis  291(%edi,%eax,4), %ecx

// CHECK: vcvttsd2usis  (%eax), %ecx
// CHECK: encoding: [0x62,0xf5,0x7f,0x08,0x6c,0x08]
          vcvttsd2usis  (%eax), %ecx

// CHECK: vcvttsd2usis  -256(,%ebp,2), %ecx
// CHECK: encoding: [0x62,0xf5,0x7f,0x08,0x6c,0x0c,0x6d,0x00,0xff,0xff,0xff]
          vcvttsd2usis  -256(,%ebp,2), %ecx

// CHECK: vcvttsd2usis  1016(%ecx), %ecx
// CHECK: encoding: [0x62,0xf5,0x7f,0x08,0x6c,0x49,0x7f]
          vcvttsd2usis  1016(%ecx), %ecx

// CHECK: vcvttsd2usis  -1024(%edx), %ecx
// CHECK: encoding: [0x62,0xf5,0x7f,0x08,0x6c,0x4a,0x80]
          vcvttsd2usis  -1024(%edx), %ecx

// CHECK: vcvttss2sis %xmm2, %ecx
// CHECK: encoding: [0x62,0xf5,0x7e,0x08,0x6d,0xca]
          vcvttss2sis %xmm2, %ecx

// CHECK: vcvttss2sis {sae}, %xmm2, %ecx
// CHECK: encoding: [0x62,0xf5,0x7e,0x18,0x6d,0xca]
          vcvttss2sis {sae}, %xmm2, %ecx

// CHECK: vcvttss2sis  268435456(%esp,%esi,8), %ecx
// CHECK: encoding: [0x62,0xf5,0x7e,0x08,0x6d,0x8c,0xf4,0x00,0x00,0x00,0x10]
          vcvttss2sis  268435456(%esp,%esi,8), %ecx

// CHECK: vcvttss2sis  291(%edi,%eax,4), %ecx
// CHECK: encoding: [0x62,0xf5,0x7e,0x08,0x6d,0x8c,0x87,0x23,0x01,0x00,0x00]
          vcvttss2sis  291(%edi,%eax,4), %ecx

// CHECK: vcvttss2sis  (%eax), %ecx
// CHECK: encoding: [0x62,0xf5,0x7e,0x08,0x6d,0x08]
          vcvttss2sis  (%eax), %ecx

// CHECK: vcvttss2sis  -128(,%ebp,2), %ecx
// CHECK: encoding: [0x62,0xf5,0x7e,0x08,0x6d,0x0c,0x6d,0x80,0xff,0xff,0xff]
          vcvttss2sis  -128(,%ebp,2), %ecx

// CHECK: vcvttss2sis  508(%ecx), %ecx
// CHECK: encoding: [0x62,0xf5,0x7e,0x08,0x6d,0x49,0x7f]
          vcvttss2sis  508(%ecx), %ecx

// CHECK: vcvttss2sis  -512(%edx), %ecx
// CHECK: encoding: [0x62,0xf5,0x7e,0x08,0x6d,0x4a,0x80]
          vcvttss2sis  -512(%edx), %ecx

// CHECK: vcvttss2usis %xmm2, %ecx
// CHECK: encoding: [0x62,0xf5,0x7e,0x08,0x6c,0xca]
          vcvttss2usis %xmm2, %ecx

// CHECK: vcvttss2usis {sae}, %xmm2, %ecx
// CHECK: encoding: [0x62,0xf5,0x7e,0x18,0x6c,0xca]
          vcvttss2usis {sae}, %xmm2, %ecx

// CHECK: vcvttss2usis  268435456(%esp,%esi,8), %ecx
// CHECK: encoding: [0x62,0xf5,0x7e,0x08,0x6c,0x8c,0xf4,0x00,0x00,0x00,0x10]
          vcvttss2usis  268435456(%esp,%esi,8), %ecx

// CHECK: vcvttss2usis  291(%edi,%eax,4), %ecx
// CHECK: encoding: [0x62,0xf5,0x7e,0x08,0x6c,0x8c,0x87,0x23,0x01,0x00,0x00]
          vcvttss2usis  291(%edi,%eax,4), %ecx

// CHECK: vcvttss2usis  (%eax), %ecx
// CHECK: encoding: [0x62,0xf5,0x7e,0x08,0x6c,0x08]
          vcvttss2usis  (%eax), %ecx

// CHECK: vcvttss2usis  -128(,%ebp,2), %ecx
// CHECK: encoding: [0x62,0xf5,0x7e,0x08,0x6c,0x0c,0x6d,0x80,0xff,0xff,0xff]
          vcvttss2usis  -128(,%ebp,2), %ecx

// CHECK: vcvttss2usis  508(%ecx), %ecx
// CHECK: encoding: [0x62,0xf5,0x7e,0x08,0x6c,0x49,0x7f]
          vcvttss2usis  508(%ecx), %ecx

// CHECK: vcvttss2usis  -512(%edx), %ecx
// CHECK: encoding: [0x62,0xf5,0x7e,0x08,0x6c,0x4a,0x80]
          vcvttss2usis  -512(%edx), %ecx

// CHECK: vcvttpd2dqs %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf5,0xfc,0x08,0x6d,0xd3]
          vcvttpd2dqs %xmm3, %xmm2

// CHECK: vcvttpd2dqs %xmm3, %xmm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0xfc,0x0f,0x6d,0xd3]
          vcvttpd2dqs %xmm3, %xmm2 {%k7}

// CHECK: vcvttpd2dqs %xmm3, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0xfc,0x8f,0x6d,0xd3]
          vcvttpd2dqs %xmm3, %xmm2 {%k7} {z}

// CHECK: vcvttpd2dqs %ymm3, %xmm2
// CHECK: encoding: [0x62,0xf5,0xfc,0x28,0x6d,0xd3]
          vcvttpd2dqs %ymm3, %xmm2

// CHECK: vcvttpd2dqs {sae}, %ymm3, %xmm2
// CHECK: encoding: [0x62,0xf5,0xf8,0x18,0x6d,0xd3]
          vcvttpd2dqs {sae}, %ymm3, %xmm2

// CHECK: vcvttpd2dqs %ymm3, %xmm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0xfc,0x2f,0x6d,0xd3]
          vcvttpd2dqs %ymm3, %xmm2 {%k7}

// CHECK: vcvttpd2dqs {sae}, %ymm3, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0xf8,0x9f,0x6d,0xd3]
          vcvttpd2dqs {sae}, %ymm3, %xmm2 {%k7} {z}

// CHECK: vcvttpd2dqs %zmm3, %ymm2
// CHECK: encoding: [0x62,0xf5,0xfc,0x48,0x6d,0xd3]
          vcvttpd2dqs %zmm3, %ymm2

// CHECK: vcvttpd2dqs {sae}, %zmm3, %ymm2
// CHECK: encoding: [0x62,0xf5,0xfc,0x18,0x6d,0xd3]
          vcvttpd2dqs {sae}, %zmm3, %ymm2

// CHECK: vcvttpd2dqs %zmm3, %ymm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0xfc,0x4f,0x6d,0xd3]
          vcvttpd2dqs %zmm3, %ymm2 {%k7}

// CHECK: vcvttpd2dqs {sae}, %zmm3, %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0xfc,0x9f,0x6d,0xd3]
          vcvttpd2dqs {sae}, %zmm3, %ymm2 {%k7} {z}

// CHECK: vcvttpd2dqsx  268435456(%esp,%esi,8), %xmm2
// CHECK: encoding: [0x62,0xf5,0xfc,0x08,0x6d,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvttpd2dqsx  268435456(%esp,%esi,8), %xmm2

// CHECK: vcvttpd2dqsx  291(%edi,%eax,4), %xmm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0xfc,0x0f,0x6d,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvttpd2dqsx  291(%edi,%eax,4), %xmm2 {%k7}

// CHECK: vcvttpd2dqs  (%eax){1to2}, %xmm2
// CHECK: encoding: [0x62,0xf5,0xfc,0x18,0x6d,0x10]
          vcvttpd2dqs  (%eax){1to2}, %xmm2

// CHECK: vcvttpd2dqsx  -512(,%ebp,2), %xmm2
// CHECK: encoding: [0x62,0xf5,0xfc,0x08,0x6d,0x14,0x6d,0x00,0xfe,0xff,0xff]
          vcvttpd2dqsx  -512(,%ebp,2), %xmm2

// CHECK: vcvttpd2dqsx  2032(%ecx), %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0xfc,0x8f,0x6d,0x51,0x7f]
          vcvttpd2dqsx  2032(%ecx), %xmm2 {%k7} {z}

// CHECK: vcvttpd2dqs  -1024(%edx){1to2}, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0xfc,0x9f,0x6d,0x52,0x80]
          vcvttpd2dqs  -1024(%edx){1to2}, %xmm2 {%k7} {z}

// CHECK: vcvttpd2dqs  (%eax){1to4}, %xmm2
// CHECK: encoding: [0x62,0xf5,0xfc,0x38,0x6d,0x10]
          vcvttpd2dqs  (%eax){1to4}, %xmm2

// CHECK: vcvttpd2dqsy  -1024(,%ebp,2), %xmm2
// CHECK: encoding: [0x62,0xf5,0xfc,0x28,0x6d,0x14,0x6d,0x00,0xfc,0xff,0xff]
          vcvttpd2dqsy  -1024(,%ebp,2), %xmm2

// CHECK: vcvttpd2dqsy  4064(%ecx), %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0xfc,0xaf,0x6d,0x51,0x7f]
          vcvttpd2dqsy  4064(%ecx), %xmm2 {%k7} {z}

// CHECK: vcvttpd2dqs  -1024(%edx){1to4}, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0xfc,0xbf,0x6d,0x52,0x80]
          vcvttpd2dqs  -1024(%edx){1to4}, %xmm2 {%k7} {z}

// CHECK: vcvttpd2dqs  268435456(%esp,%esi,8), %ymm2
// CHECK: encoding: [0x62,0xf5,0xfc,0x48,0x6d,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvttpd2dqs  268435456(%esp,%esi,8), %ymm2

// CHECK: vcvttpd2dqs  291(%edi,%eax,4), %ymm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0xfc,0x4f,0x6d,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvttpd2dqs  291(%edi,%eax,4), %ymm2 {%k7}

// CHECK: vcvttpd2dqs  (%eax){1to8}, %ymm2
// CHECK: encoding: [0x62,0xf5,0xfc,0x58,0x6d,0x10]
          vcvttpd2dqs  (%eax){1to8}, %ymm2

// CHECK: vcvttpd2dqs  -2048(,%ebp,2), %ymm2
// CHECK: encoding: [0x62,0xf5,0xfc,0x48,0x6d,0x14,0x6d,0x00,0xf8,0xff,0xff]
          vcvttpd2dqs  -2048(,%ebp,2), %ymm2

// CHECK: vcvttpd2dqs  8128(%ecx), %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0xfc,0xcf,0x6d,0x51,0x7f]
          vcvttpd2dqs  8128(%ecx), %ymm2 {%k7} {z}

// CHECK: vcvttpd2dqs  -1024(%edx){1to8}, %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0xfc,0xdf,0x6d,0x52,0x80]
          vcvttpd2dqs  -1024(%edx){1to8}, %ymm2 {%k7} {z}

// CHECK: vcvttpd2qqs %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf5,0xfd,0x08,0x6d,0xd3]
          vcvttpd2qqs %xmm3, %xmm2

// CHECK: vcvttpd2qqs %xmm3, %xmm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0xfd,0x0f,0x6d,0xd3]
          vcvttpd2qqs %xmm3, %xmm2 {%k7}

// CHECK: vcvttpd2qqs %xmm3, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0xfd,0x8f,0x6d,0xd3]
          vcvttpd2qqs %xmm3, %xmm2 {%k7} {z}

// CHECK: vcvttpd2qqs %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf5,0xfd,0x28,0x6d,0xd3]
          vcvttpd2qqs %ymm3, %ymm2

// CHECK: vcvttpd2qqs {sae}, %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf5,0xf9,0x18,0x6d,0xd3]
          vcvttpd2qqs {sae}, %ymm3, %ymm2

// CHECK: vcvttpd2qqs %ymm3, %ymm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0xfd,0x2f,0x6d,0xd3]
          vcvttpd2qqs %ymm3, %ymm2 {%k7}

// CHECK: vcvttpd2qqs {sae}, %ymm3, %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0xf9,0x9f,0x6d,0xd3]
          vcvttpd2qqs {sae}, %ymm3, %ymm2 {%k7} {z}

// CHECK: vcvttpd2qqs %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf5,0xfd,0x48,0x6d,0xd3]
          vcvttpd2qqs %zmm3, %zmm2

// CHECK: vcvttpd2qqs {sae}, %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf5,0xfd,0x18,0x6d,0xd3]
          vcvttpd2qqs {sae}, %zmm3, %zmm2

// CHECK: vcvttpd2qqs %zmm3, %zmm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0xfd,0x4f,0x6d,0xd3]
          vcvttpd2qqs %zmm3, %zmm2 {%k7}

// CHECK: vcvttpd2qqs {sae}, %zmm3, %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0xfd,0x9f,0x6d,0xd3]
          vcvttpd2qqs {sae}, %zmm3, %zmm2 {%k7} {z}

// CHECK: vcvttpd2qqs  268435456(%esp,%esi,8), %xmm2
// CHECK: encoding: [0x62,0xf5,0xfd,0x08,0x6d,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvttpd2qqs  268435456(%esp,%esi,8), %xmm2

// CHECK: vcvttpd2qqs  291(%edi,%eax,4), %xmm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0xfd,0x0f,0x6d,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvttpd2qqs  291(%edi,%eax,4), %xmm2 {%k7}

// CHECK: vcvttpd2qqs  (%eax){1to2}, %xmm2
// CHECK: encoding: [0x62,0xf5,0xfd,0x18,0x6d,0x10]
          vcvttpd2qqs  (%eax){1to2}, %xmm2

// CHECK: vcvttpd2qqs  -512(,%ebp,2), %xmm2
// CHECK: encoding: [0x62,0xf5,0xfd,0x08,0x6d,0x14,0x6d,0x00,0xfe,0xff,0xff]
          vcvttpd2qqs  -512(,%ebp,2), %xmm2

// CHECK: vcvttpd2qqs  2032(%ecx), %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0xfd,0x8f,0x6d,0x51,0x7f]
          vcvttpd2qqs  2032(%ecx), %xmm2 {%k7} {z}

// CHECK: vcvttpd2qqs  -1024(%edx){1to2}, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0xfd,0x9f,0x6d,0x52,0x80]
          vcvttpd2qqs  -1024(%edx){1to2}, %xmm2 {%k7} {z}

// CHECK: vcvttpd2qqs  268435456(%esp,%esi,8), %ymm2
// CHECK: encoding: [0x62,0xf5,0xfd,0x28,0x6d,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvttpd2qqs  268435456(%esp,%esi,8), %ymm2

// CHECK: vcvttpd2qqs  291(%edi,%eax,4), %ymm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0xfd,0x2f,0x6d,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvttpd2qqs  291(%edi,%eax,4), %ymm2 {%k7}

// CHECK: vcvttpd2qqs  (%eax){1to4}, %ymm2
// CHECK: encoding: [0x62,0xf5,0xfd,0x38,0x6d,0x10]
          vcvttpd2qqs  (%eax){1to4}, %ymm2

// CHECK: vcvttpd2qqs  -1024(,%ebp,2), %ymm2
// CHECK: encoding: [0x62,0xf5,0xfd,0x28,0x6d,0x14,0x6d,0x00,0xfc,0xff,0xff]
          vcvttpd2qqs  -1024(,%ebp,2), %ymm2

// CHECK: vcvttpd2qqs  4064(%ecx), %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0xfd,0xaf,0x6d,0x51,0x7f]
          vcvttpd2qqs  4064(%ecx), %ymm2 {%k7} {z}

// CHECK: vcvttpd2qqs  -1024(%edx){1to4}, %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0xfd,0xbf,0x6d,0x52,0x80]
          vcvttpd2qqs  -1024(%edx){1to4}, %ymm2 {%k7} {z}

// CHECK: vcvttpd2qqs  268435456(%esp,%esi,8), %zmm2
// CHECK: encoding: [0x62,0xf5,0xfd,0x48,0x6d,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvttpd2qqs  268435456(%esp,%esi,8), %zmm2

// CHECK: vcvttpd2qqs  291(%edi,%eax,4), %zmm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0xfd,0x4f,0x6d,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvttpd2qqs  291(%edi,%eax,4), %zmm2 {%k7}

// CHECK: vcvttpd2qqs  (%eax){1to8}, %zmm2
// CHECK: encoding: [0x62,0xf5,0xfd,0x58,0x6d,0x10]
          vcvttpd2qqs  (%eax){1to8}, %zmm2

// CHECK: vcvttpd2qqs  -2048(,%ebp,2), %zmm2
// CHECK: encoding: [0x62,0xf5,0xfd,0x48,0x6d,0x14,0x6d,0x00,0xf8,0xff,0xff]
          vcvttpd2qqs  -2048(,%ebp,2), %zmm2

// CHECK: vcvttpd2qqs  8128(%ecx), %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0xfd,0xcf,0x6d,0x51,0x7f]
          vcvttpd2qqs  8128(%ecx), %zmm2 {%k7} {z}

// CHECK: vcvttpd2qqs  -1024(%edx){1to8}, %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0xfd,0xdf,0x6d,0x52,0x80]
          vcvttpd2qqs  -1024(%edx){1to8}, %zmm2 {%k7} {z}

// CHECK: vcvttpd2udqs %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf5,0xfc,0x08,0x6c,0xd3]
          vcvttpd2udqs %xmm3, %xmm2

// CHECK: vcvttpd2udqs %xmm3, %xmm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0xfc,0x0f,0x6c,0xd3]
          vcvttpd2udqs %xmm3, %xmm2 {%k7}

// CHECK: vcvttpd2udqs %xmm3, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0xfc,0x8f,0x6c,0xd3]
          vcvttpd2udqs %xmm3, %xmm2 {%k7} {z}

// CHECK: vcvttpd2udqs %ymm3, %xmm2
// CHECK: encoding: [0x62,0xf5,0xfc,0x28,0x6c,0xd3]
          vcvttpd2udqs %ymm3, %xmm2

// CHECK: vcvttpd2udqs {sae}, %ymm3, %xmm2
// CHECK: encoding: [0x62,0xf5,0xf8,0x18,0x6c,0xd3]
          vcvttpd2udqs {sae}, %ymm3, %xmm2

// CHECK: vcvttpd2udqs %ymm3, %xmm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0xfc,0x2f,0x6c,0xd3]
          vcvttpd2udqs %ymm3, %xmm2 {%k7}

// CHECK: vcvttpd2udqs {sae}, %ymm3, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0xf8,0x9f,0x6c,0xd3]
          vcvttpd2udqs {sae}, %ymm3, %xmm2 {%k7} {z}

// CHECK: vcvttpd2udqs %zmm3, %ymm2
// CHECK: encoding: [0x62,0xf5,0xfc,0x48,0x6c,0xd3]
          vcvttpd2udqs %zmm3, %ymm2

// CHECK: vcvttpd2udqs {sae}, %zmm3, %ymm2
// CHECK: encoding: [0x62,0xf5,0xfc,0x18,0x6c,0xd3]
          vcvttpd2udqs {sae}, %zmm3, %ymm2

// CHECK: vcvttpd2udqs %zmm3, %ymm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0xfc,0x4f,0x6c,0xd3]
          vcvttpd2udqs %zmm3, %ymm2 {%k7}

// CHECK: vcvttpd2udqs {sae}, %zmm3, %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0xfc,0x9f,0x6c,0xd3]
          vcvttpd2udqs {sae}, %zmm3, %ymm2 {%k7} {z}

// CHECK: vcvttpd2udqsx  268435456(%esp,%esi,8), %xmm2
// CHECK: encoding: [0x62,0xf5,0xfc,0x08,0x6c,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvttpd2udqsx  268435456(%esp,%esi,8), %xmm2

// CHECK: vcvttpd2udqsx  291(%edi,%eax,4), %xmm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0xfc,0x0f,0x6c,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvttpd2udqsx  291(%edi,%eax,4), %xmm2 {%k7}

// CHECK: vcvttpd2udqs  (%eax){1to2}, %xmm2
// CHECK: encoding: [0x62,0xf5,0xfc,0x18,0x6c,0x10]
          vcvttpd2udqs  (%eax){1to2}, %xmm2

// CHECK: vcvttpd2udqsx  -512(,%ebp,2), %xmm2
// CHECK: encoding: [0x62,0xf5,0xfc,0x08,0x6c,0x14,0x6d,0x00,0xfe,0xff,0xff]
          vcvttpd2udqsx  -512(,%ebp,2), %xmm2

// CHECK: vcvttpd2udqsx  2032(%ecx), %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0xfc,0x8f,0x6c,0x51,0x7f]
          vcvttpd2udqsx  2032(%ecx), %xmm2 {%k7} {z}

// CHECK: vcvttpd2udqs  -1024(%edx){1to2}, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0xfc,0x9f,0x6c,0x52,0x80]
          vcvttpd2udqs  -1024(%edx){1to2}, %xmm2 {%k7} {z}

// CHECK: vcvttpd2udqs  (%eax){1to4}, %xmm2
// CHECK: encoding: [0x62,0xf5,0xfc,0x38,0x6c,0x10]
          vcvttpd2udqs  (%eax){1to4}, %xmm2

// CHECK: vcvttpd2udqsy  -1024(,%ebp,2), %xmm2
// CHECK: encoding: [0x62,0xf5,0xfc,0x28,0x6c,0x14,0x6d,0x00,0xfc,0xff,0xff]
          vcvttpd2udqsy  -1024(,%ebp,2), %xmm2

// CHECK: vcvttpd2udqsy  4064(%ecx), %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0xfc,0xaf,0x6c,0x51,0x7f]
          vcvttpd2udqsy  4064(%ecx), %xmm2 {%k7} {z}

// CHECK: vcvttpd2udqs  -1024(%edx){1to4}, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0xfc,0xbf,0x6c,0x52,0x80]
          vcvttpd2udqs  -1024(%edx){1to4}, %xmm2 {%k7} {z}

// CHECK: vcvttpd2udqs  268435456(%esp,%esi,8), %ymm2
// CHECK: encoding: [0x62,0xf5,0xfc,0x48,0x6c,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvttpd2udqs  268435456(%esp,%esi,8), %ymm2

// CHECK: vcvttpd2udqs  291(%edi,%eax,4), %ymm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0xfc,0x4f,0x6c,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvttpd2udqs  291(%edi,%eax,4), %ymm2 {%k7}

// CHECK: vcvttpd2udqs  (%eax){1to8}, %ymm2
// CHECK: encoding: [0x62,0xf5,0xfc,0x58,0x6c,0x10]
          vcvttpd2udqs  (%eax){1to8}, %ymm2

// CHECK: vcvttpd2udqs  -2048(,%ebp,2), %ymm2
// CHECK: encoding: [0x62,0xf5,0xfc,0x48,0x6c,0x14,0x6d,0x00,0xf8,0xff,0xff]
          vcvttpd2udqs  -2048(,%ebp,2), %ymm2

// CHECK: vcvttpd2udqs  8128(%ecx), %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0xfc,0xcf,0x6c,0x51,0x7f]
          vcvttpd2udqs  8128(%ecx), %ymm2 {%k7} {z}

// CHECK: vcvttpd2udqs  -1024(%edx){1to8}, %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0xfc,0xdf,0x6c,0x52,0x80]
          vcvttpd2udqs  -1024(%edx){1to8}, %ymm2 {%k7} {z}

// CHECK: vcvttpd2uqqs %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf5,0xfd,0x08,0x6c,0xd3]
          vcvttpd2uqqs %xmm3, %xmm2

// CHECK: vcvttpd2uqqs %xmm3, %xmm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0xfd,0x0f,0x6c,0xd3]
          vcvttpd2uqqs %xmm3, %xmm2 {%k7}

// CHECK: vcvttpd2uqqs %xmm3, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0xfd,0x8f,0x6c,0xd3]
          vcvttpd2uqqs %xmm3, %xmm2 {%k7} {z}

// CHECK: vcvttpd2uqqs %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf5,0xfd,0x28,0x6c,0xd3]
          vcvttpd2uqqs %ymm3, %ymm2

// CHECK: vcvttpd2uqqs {sae}, %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf5,0xf9,0x18,0x6c,0xd3]
          vcvttpd2uqqs {sae}, %ymm3, %ymm2

// CHECK: vcvttpd2uqqs %ymm3, %ymm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0xfd,0x2f,0x6c,0xd3]
          vcvttpd2uqqs %ymm3, %ymm2 {%k7}

// CHECK: vcvttpd2uqqs {sae}, %ymm3, %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0xf9,0x9f,0x6c,0xd3]
          vcvttpd2uqqs {sae}, %ymm3, %ymm2 {%k7} {z}

// CHECK: vcvttpd2uqqs %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf5,0xfd,0x48,0x6c,0xd3]
          vcvttpd2uqqs %zmm3, %zmm2

// CHECK: vcvttpd2uqqs {sae}, %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf5,0xfd,0x18,0x6c,0xd3]
          vcvttpd2uqqs {sae}, %zmm3, %zmm2

// CHECK: vcvttpd2uqqs %zmm3, %zmm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0xfd,0x4f,0x6c,0xd3]
          vcvttpd2uqqs %zmm3, %zmm2 {%k7}

// CHECK: vcvttpd2uqqs {sae}, %zmm3, %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0xfd,0x9f,0x6c,0xd3]
          vcvttpd2uqqs {sae}, %zmm3, %zmm2 {%k7} {z}

// CHECK: vcvttpd2uqqs  268435456(%esp,%esi,8), %xmm2
// CHECK: encoding: [0x62,0xf5,0xfd,0x08,0x6c,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvttpd2uqqs  268435456(%esp,%esi,8), %xmm2

// CHECK: vcvttpd2uqqs  291(%edi,%eax,4), %xmm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0xfd,0x0f,0x6c,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvttpd2uqqs  291(%edi,%eax,4), %xmm2 {%k7}

// CHECK: vcvttpd2uqqs  (%eax){1to2}, %xmm2
// CHECK: encoding: [0x62,0xf5,0xfd,0x18,0x6c,0x10]
          vcvttpd2uqqs  (%eax){1to2}, %xmm2

// CHECK: vcvttpd2uqqs  -512(,%ebp,2), %xmm2
// CHECK: encoding: [0x62,0xf5,0xfd,0x08,0x6c,0x14,0x6d,0x00,0xfe,0xff,0xff]
          vcvttpd2uqqs  -512(,%ebp,2), %xmm2

// CHECK: vcvttpd2uqqs  2032(%ecx), %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0xfd,0x8f,0x6c,0x51,0x7f]
          vcvttpd2uqqs  2032(%ecx), %xmm2 {%k7} {z}

// CHECK: vcvttpd2uqqs  -1024(%edx){1to2}, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0xfd,0x9f,0x6c,0x52,0x80]
          vcvttpd2uqqs  -1024(%edx){1to2}, %xmm2 {%k7} {z}

// CHECK: vcvttpd2uqqs  268435456(%esp,%esi,8), %ymm2
// CHECK: encoding: [0x62,0xf5,0xfd,0x28,0x6c,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvttpd2uqqs  268435456(%esp,%esi,8), %ymm2

// CHECK: vcvttpd2uqqs  291(%edi,%eax,4), %ymm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0xfd,0x2f,0x6c,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvttpd2uqqs  291(%edi,%eax,4), %ymm2 {%k7}

// CHECK: vcvttpd2uqqs  (%eax){1to4}, %ymm2
// CHECK: encoding: [0x62,0xf5,0xfd,0x38,0x6c,0x10]
          vcvttpd2uqqs  (%eax){1to4}, %ymm2

// CHECK: vcvttpd2uqqs  -1024(,%ebp,2), %ymm2
// CHECK: encoding: [0x62,0xf5,0xfd,0x28,0x6c,0x14,0x6d,0x00,0xfc,0xff,0xff]
          vcvttpd2uqqs  -1024(,%ebp,2), %ymm2

// CHECK: vcvttpd2uqqs  4064(%ecx), %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0xfd,0xaf,0x6c,0x51,0x7f]
          vcvttpd2uqqs  4064(%ecx), %ymm2 {%k7} {z}

// CHECK: vcvttpd2uqqs  -1024(%edx){1to4}, %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0xfd,0xbf,0x6c,0x52,0x80]
          vcvttpd2uqqs  -1024(%edx){1to4}, %ymm2 {%k7} {z}

// CHECK: vcvttpd2uqqs  268435456(%esp,%esi,8), %zmm2
// CHECK: encoding: [0x62,0xf5,0xfd,0x48,0x6c,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvttpd2uqqs  268435456(%esp,%esi,8), %zmm2

// CHECK: vcvttpd2uqqs  291(%edi,%eax,4), %zmm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0xfd,0x4f,0x6c,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvttpd2uqqs  291(%edi,%eax,4), %zmm2 {%k7}

// CHECK: vcvttpd2uqqs  (%eax){1to8}, %zmm2
// CHECK: encoding: [0x62,0xf5,0xfd,0x58,0x6c,0x10]
          vcvttpd2uqqs  (%eax){1to8}, %zmm2

// CHECK: vcvttpd2uqqs  -2048(,%ebp,2), %zmm2
// CHECK: encoding: [0x62,0xf5,0xfd,0x48,0x6c,0x14,0x6d,0x00,0xf8,0xff,0xff]
          vcvttpd2uqqs  -2048(,%ebp,2), %zmm2

// CHECK: vcvttpd2uqqs  8128(%ecx), %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0xfd,0xcf,0x6c,0x51,0x7f]
          vcvttpd2uqqs  8128(%ecx), %zmm2 {%k7} {z}

// CHECK: vcvttpd2uqqs  -1024(%edx){1to8}, %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0xfd,0xdf,0x6c,0x52,0x80]
          vcvttpd2uqqs  -1024(%edx){1to8}, %zmm2 {%k7} {z}

// CHECK: vcvttps2dqs %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf5,0x7c,0x08,0x6d,0xd3]
          vcvttps2dqs %xmm3, %xmm2

// CHECK: vcvttps2dqs %xmm3, %xmm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7c,0x0f,0x6d,0xd3]
          vcvttps2dqs %xmm3, %xmm2 {%k7}

// CHECK: vcvttps2dqs %xmm3, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7c,0x8f,0x6d,0xd3]
          vcvttps2dqs %xmm3, %xmm2 {%k7} {z}

// CHECK: vcvttps2dqs %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf5,0x7c,0x28,0x6d,0xd3]
          vcvttps2dqs %ymm3, %ymm2

// CHECK: vcvttps2dqs {sae}, %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf5,0x78,0x18,0x6d,0xd3]
          vcvttps2dqs {sae}, %ymm3, %ymm2

// CHECK: vcvttps2dqs %ymm3, %ymm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7c,0x2f,0x6d,0xd3]
          vcvttps2dqs %ymm3, %ymm2 {%k7}

// CHECK: vcvttps2dqs {sae}, %ymm3, %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x78,0x9f,0x6d,0xd3]
          vcvttps2dqs {sae}, %ymm3, %ymm2 {%k7} {z}

// CHECK: vcvttps2dqs %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf5,0x7c,0x48,0x6d,0xd3]
          vcvttps2dqs %zmm3, %zmm2

// CHECK: vcvttps2dqs {sae}, %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf5,0x7c,0x18,0x6d,0xd3]
          vcvttps2dqs {sae}, %zmm3, %zmm2

// CHECK: vcvttps2dqs %zmm3, %zmm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7c,0x4f,0x6d,0xd3]
          vcvttps2dqs %zmm3, %zmm2 {%k7}

// CHECK: vcvttps2dqs {sae}, %zmm3, %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7c,0x9f,0x6d,0xd3]
          vcvttps2dqs {sae}, %zmm3, %zmm2 {%k7} {z}

// CHECK: vcvttps2dqs  268435456(%esp,%esi,8), %xmm2
// CHECK: encoding: [0x62,0xf5,0x7c,0x08,0x6d,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvttps2dqs  268435456(%esp,%esi,8), %xmm2

// CHECK: vcvttps2dqs  291(%edi,%eax,4), %xmm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7c,0x0f,0x6d,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvttps2dqs  291(%edi,%eax,4), %xmm2 {%k7}

// CHECK: vcvttps2dqs  (%eax){1to4}, %xmm2
// CHECK: encoding: [0x62,0xf5,0x7c,0x18,0x6d,0x10]
          vcvttps2dqs  (%eax){1to4}, %xmm2

// CHECK: vcvttps2dqs  -512(,%ebp,2), %xmm2
// CHECK: encoding: [0x62,0xf5,0x7c,0x08,0x6d,0x14,0x6d,0x00,0xfe,0xff,0xff]
          vcvttps2dqs  -512(,%ebp,2), %xmm2

// CHECK: vcvttps2dqs  2032(%ecx), %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7c,0x8f,0x6d,0x51,0x7f]
          vcvttps2dqs  2032(%ecx), %xmm2 {%k7} {z}

// CHECK: vcvttps2dqs  -512(%edx){1to4}, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7c,0x9f,0x6d,0x52,0x80]
          vcvttps2dqs  -512(%edx){1to4}, %xmm2 {%k7} {z}

// CHECK: vcvttps2dqs  268435456(%esp,%esi,8), %ymm2
// CHECK: encoding: [0x62,0xf5,0x7c,0x28,0x6d,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvttps2dqs  268435456(%esp,%esi,8), %ymm2

// CHECK: vcvttps2dqs  291(%edi,%eax,4), %ymm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7c,0x2f,0x6d,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvttps2dqs  291(%edi,%eax,4), %ymm2 {%k7}

// CHECK: vcvttps2dqs  (%eax){1to8}, %ymm2
// CHECK: encoding: [0x62,0xf5,0x7c,0x38,0x6d,0x10]
          vcvttps2dqs  (%eax){1to8}, %ymm2

// CHECK: vcvttps2dqs  -1024(,%ebp,2), %ymm2
// CHECK: encoding: [0x62,0xf5,0x7c,0x28,0x6d,0x14,0x6d,0x00,0xfc,0xff,0xff]
          vcvttps2dqs  -1024(,%ebp,2), %ymm2

// CHECK: vcvttps2dqs  4064(%ecx), %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7c,0xaf,0x6d,0x51,0x7f]
          vcvttps2dqs  4064(%ecx), %ymm2 {%k7} {z}

// CHECK: vcvttps2dqs  -512(%edx){1to8}, %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7c,0xbf,0x6d,0x52,0x80]
          vcvttps2dqs  -512(%edx){1to8}, %ymm2 {%k7} {z}

// CHECK: vcvttps2dqs  268435456(%esp,%esi,8), %zmm2
// CHECK: encoding: [0x62,0xf5,0x7c,0x48,0x6d,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvttps2dqs  268435456(%esp,%esi,8), %zmm2

// CHECK: vcvttps2dqs  291(%edi,%eax,4), %zmm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7c,0x4f,0x6d,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvttps2dqs  291(%edi,%eax,4), %zmm2 {%k7}

// CHECK: vcvttps2dqs  (%eax){1to16}, %zmm2
// CHECK: encoding: [0x62,0xf5,0x7c,0x58,0x6d,0x10]
          vcvttps2dqs  (%eax){1to16}, %zmm2

// CHECK: vcvttps2dqs  -2048(,%ebp,2), %zmm2
// CHECK: encoding: [0x62,0xf5,0x7c,0x48,0x6d,0x14,0x6d,0x00,0xf8,0xff,0xff]
          vcvttps2dqs  -2048(,%ebp,2), %zmm2

// CHECK: vcvttps2dqs  8128(%ecx), %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7c,0xcf,0x6d,0x51,0x7f]
          vcvttps2dqs  8128(%ecx), %zmm2 {%k7} {z}

// CHECK: vcvttps2dqs  -512(%edx){1to16}, %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7c,0xdf,0x6d,0x52,0x80]
          vcvttps2dqs  -512(%edx){1to16}, %zmm2 {%k7} {z}

// CHECK: vcvttps2qqs %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf5,0x7d,0x08,0x6d,0xd3]
          vcvttps2qqs %xmm3, %xmm2

// CHECK: vcvttps2qqs %xmm3, %xmm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7d,0x0f,0x6d,0xd3]
          vcvttps2qqs %xmm3, %xmm2 {%k7}

// CHECK: vcvttps2qqs %xmm3, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7d,0x8f,0x6d,0xd3]
          vcvttps2qqs %xmm3, %xmm2 {%k7} {z}

// CHECK: vcvttps2qqs %xmm3, %ymm2
// CHECK: encoding: [0x62,0xf5,0x7d,0x28,0x6d,0xd3]
          vcvttps2qqs %xmm3, %ymm2

// CHECK: vcvttps2qqs {sae}, %xmm3, %ymm2
// CHECK: encoding: [0x62,0xf5,0x79,0x18,0x6d,0xd3]
          vcvttps2qqs {sae}, %xmm3, %ymm2

// CHECK: vcvttps2qqs %xmm3, %ymm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7d,0x2f,0x6d,0xd3]
          vcvttps2qqs %xmm3, %ymm2 {%k7}

// CHECK: vcvttps2qqs {sae}, %xmm3, %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x79,0x9f,0x6d,0xd3]
          vcvttps2qqs {sae}, %xmm3, %ymm2 {%k7} {z}

// CHECK: vcvttps2qqs %ymm3, %zmm2
// CHECK: encoding: [0x62,0xf5,0x7d,0x48,0x6d,0xd3]
          vcvttps2qqs %ymm3, %zmm2

// CHECK: vcvttps2qqs {sae}, %ymm3, %zmm2
// CHECK: encoding: [0x62,0xf5,0x7d,0x18,0x6d,0xd3]
          vcvttps2qqs {sae}, %ymm3, %zmm2

// CHECK: vcvttps2qqs %ymm3, %zmm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7d,0x4f,0x6d,0xd3]
          vcvttps2qqs %ymm3, %zmm2 {%k7}

// CHECK: vcvttps2qqs {sae}, %ymm3, %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7d,0x9f,0x6d,0xd3]
          vcvttps2qqs {sae}, %ymm3, %zmm2 {%k7} {z}

// CHECK: vcvttps2qqs  268435456(%esp,%esi,8), %xmm2
// CHECK: encoding: [0x62,0xf5,0x7d,0x08,0x6d,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvttps2qqs  268435456(%esp,%esi,8), %xmm2

// CHECK: vcvttps2qqs  291(%edi,%eax,4), %xmm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7d,0x0f,0x6d,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvttps2qqs  291(%edi,%eax,4), %xmm2 {%k7}

// CHECK: vcvttps2qqs  (%eax){1to2}, %xmm2
// CHECK: encoding: [0x62,0xf5,0x7d,0x18,0x6d,0x10]
          vcvttps2qqs  (%eax){1to2}, %xmm2

// CHECK: vcvttps2qqs  -256(,%ebp,2), %xmm2
// CHECK: encoding: [0x62,0xf5,0x7d,0x08,0x6d,0x14,0x6d,0x00,0xff,0xff,0xff]
          vcvttps2qqs  -256(,%ebp,2), %xmm2

// CHECK: vcvttps2qqs  1016(%ecx), %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7d,0x8f,0x6d,0x51,0x7f]
          vcvttps2qqs  1016(%ecx), %xmm2 {%k7} {z}

// CHECK: vcvttps2qqs  -512(%edx){1to2}, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7d,0x9f,0x6d,0x52,0x80]
          vcvttps2qqs  -512(%edx){1to2}, %xmm2 {%k7} {z}

// CHECK: vcvttps2qqs  268435456(%esp,%esi,8), %ymm2
// CHECK: encoding: [0x62,0xf5,0x7d,0x28,0x6d,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvttps2qqs  268435456(%esp,%esi,8), %ymm2

// CHECK: vcvttps2qqs  291(%edi,%eax,4), %ymm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7d,0x2f,0x6d,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvttps2qqs  291(%edi,%eax,4), %ymm2 {%k7}

// CHECK: vcvttps2qqs  (%eax){1to4}, %ymm2
// CHECK: encoding: [0x62,0xf5,0x7d,0x38,0x6d,0x10]
          vcvttps2qqs  (%eax){1to4}, %ymm2

// CHECK: vcvttps2qqs  -512(,%ebp,2), %ymm2
// CHECK: encoding: [0x62,0xf5,0x7d,0x28,0x6d,0x14,0x6d,0x00,0xfe,0xff,0xff]
          vcvttps2qqs  -512(,%ebp,2), %ymm2

// CHECK: vcvttps2qqs  2032(%ecx), %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7d,0xaf,0x6d,0x51,0x7f]
          vcvttps2qqs  2032(%ecx), %ymm2 {%k7} {z}

// CHECK: vcvttps2qqs  -512(%edx){1to4}, %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7d,0xbf,0x6d,0x52,0x80]
          vcvttps2qqs  -512(%edx){1to4}, %ymm2 {%k7} {z}

// CHECK: vcvttps2qqs  268435456(%esp,%esi,8), %zmm2
// CHECK: encoding: [0x62,0xf5,0x7d,0x48,0x6d,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvttps2qqs  268435456(%esp,%esi,8), %zmm2

// CHECK: vcvttps2qqs  291(%edi,%eax,4), %zmm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7d,0x4f,0x6d,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvttps2qqs  291(%edi,%eax,4), %zmm2 {%k7}

// CHECK: vcvttps2qqs  (%eax){1to8}, %zmm2
// CHECK: encoding: [0x62,0xf5,0x7d,0x58,0x6d,0x10]
          vcvttps2qqs  (%eax){1to8}, %zmm2

// CHECK: vcvttps2qqs  -1024(,%ebp,2), %zmm2
// CHECK: encoding: [0x62,0xf5,0x7d,0x48,0x6d,0x14,0x6d,0x00,0xfc,0xff,0xff]
          vcvttps2qqs  -1024(,%ebp,2), %zmm2

// CHECK: vcvttps2qqs  4064(%ecx), %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7d,0xcf,0x6d,0x51,0x7f]
          vcvttps2qqs  4064(%ecx), %zmm2 {%k7} {z}

// CHECK: vcvttps2qqs  -512(%edx){1to8}, %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7d,0xdf,0x6d,0x52,0x80]
          vcvttps2qqs  -512(%edx){1to8}, %zmm2 {%k7} {z}

// CHECK: vcvttps2udqs %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf5,0x7c,0x08,0x6c,0xd3]
          vcvttps2udqs %xmm3, %xmm2

// CHECK: vcvttps2udqs %xmm3, %xmm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7c,0x0f,0x6c,0xd3]
          vcvttps2udqs %xmm3, %xmm2 {%k7}

// CHECK: vcvttps2udqs %xmm3, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7c,0x8f,0x6c,0xd3]
          vcvttps2udqs %xmm3, %xmm2 {%k7} {z}

// CHECK: vcvttps2udqs %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf5,0x7c,0x28,0x6c,0xd3]
          vcvttps2udqs %ymm3, %ymm2

// CHECK: vcvttps2udqs {sae}, %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf5,0x78,0x18,0x6c,0xd3]
          vcvttps2udqs {sae}, %ymm3, %ymm2

// CHECK: vcvttps2udqs %ymm3, %ymm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7c,0x2f,0x6c,0xd3]
          vcvttps2udqs %ymm3, %ymm2 {%k7}

// CHECK: vcvttps2udqs {sae}, %ymm3, %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x78,0x9f,0x6c,0xd3]
          vcvttps2udqs {sae}, %ymm3, %ymm2 {%k7} {z}

// CHECK: vcvttps2udqs %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf5,0x7c,0x48,0x6c,0xd3]
          vcvttps2udqs %zmm3, %zmm2

// CHECK: vcvttps2udqs {sae}, %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf5,0x7c,0x18,0x6c,0xd3]
          vcvttps2udqs {sae}, %zmm3, %zmm2

// CHECK: vcvttps2udqs %zmm3, %zmm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7c,0x4f,0x6c,0xd3]
          vcvttps2udqs %zmm3, %zmm2 {%k7}

// CHECK: vcvttps2udqs {sae}, %zmm3, %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7c,0x9f,0x6c,0xd3]
          vcvttps2udqs {sae}, %zmm3, %zmm2 {%k7} {z}

// CHECK: vcvttps2udqs  268435456(%esp,%esi,8), %xmm2
// CHECK: encoding: [0x62,0xf5,0x7c,0x08,0x6c,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvttps2udqs  268435456(%esp,%esi,8), %xmm2

// CHECK: vcvttps2udqs  291(%edi,%eax,4), %xmm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7c,0x0f,0x6c,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvttps2udqs  291(%edi,%eax,4), %xmm2 {%k7}

// CHECK: vcvttps2udqs  (%eax){1to4}, %xmm2
// CHECK: encoding: [0x62,0xf5,0x7c,0x18,0x6c,0x10]
          vcvttps2udqs  (%eax){1to4}, %xmm2

// CHECK: vcvttps2udqs  -512(,%ebp,2), %xmm2
// CHECK: encoding: [0x62,0xf5,0x7c,0x08,0x6c,0x14,0x6d,0x00,0xfe,0xff,0xff]
          vcvttps2udqs  -512(,%ebp,2), %xmm2

// CHECK: vcvttps2udqs  2032(%ecx), %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7c,0x8f,0x6c,0x51,0x7f]
          vcvttps2udqs  2032(%ecx), %xmm2 {%k7} {z}

// CHECK: vcvttps2udqs  -512(%edx){1to4}, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7c,0x9f,0x6c,0x52,0x80]
          vcvttps2udqs  -512(%edx){1to4}, %xmm2 {%k7} {z}

// CHECK: vcvttps2udqs  268435456(%esp,%esi,8), %ymm2
// CHECK: encoding: [0x62,0xf5,0x7c,0x28,0x6c,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvttps2udqs  268435456(%esp,%esi,8), %ymm2

// CHECK: vcvttps2udqs  291(%edi,%eax,4), %ymm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7c,0x2f,0x6c,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvttps2udqs  291(%edi,%eax,4), %ymm2 {%k7}

// CHECK: vcvttps2udqs  (%eax){1to8}, %ymm2
// CHECK: encoding: [0x62,0xf5,0x7c,0x38,0x6c,0x10]
          vcvttps2udqs  (%eax){1to8}, %ymm2

// CHECK: vcvttps2udqs  -1024(,%ebp,2), %ymm2
// CHECK: encoding: [0x62,0xf5,0x7c,0x28,0x6c,0x14,0x6d,0x00,0xfc,0xff,0xff]
          vcvttps2udqs  -1024(,%ebp,2), %ymm2

// CHECK: vcvttps2udqs  4064(%ecx), %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7c,0xaf,0x6c,0x51,0x7f]
          vcvttps2udqs  4064(%ecx), %ymm2 {%k7} {z}

// CHECK: vcvttps2udqs  -512(%edx){1to8}, %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7c,0xbf,0x6c,0x52,0x80]
          vcvttps2udqs  -512(%edx){1to8}, %ymm2 {%k7} {z}

// CHECK: vcvttps2udqs  268435456(%esp,%esi,8), %zmm2
// CHECK: encoding: [0x62,0xf5,0x7c,0x48,0x6c,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvttps2udqs  268435456(%esp,%esi,8), %zmm2

// CHECK: vcvttps2udqs  291(%edi,%eax,4), %zmm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7c,0x4f,0x6c,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvttps2udqs  291(%edi,%eax,4), %zmm2 {%k7}

// CHECK: vcvttps2udqs  (%eax){1to16}, %zmm2
// CHECK: encoding: [0x62,0xf5,0x7c,0x58,0x6c,0x10]
          vcvttps2udqs  (%eax){1to16}, %zmm2

// CHECK: vcvttps2udqs  -2048(,%ebp,2), %zmm2
// CHECK: encoding: [0x62,0xf5,0x7c,0x48,0x6c,0x14,0x6d,0x00,0xf8,0xff,0xff]
          vcvttps2udqs  -2048(,%ebp,2), %zmm2

// CHECK: vcvttps2udqs  8128(%ecx), %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7c,0xcf,0x6c,0x51,0x7f]
          vcvttps2udqs  8128(%ecx), %zmm2 {%k7} {z}

// CHECK: vcvttps2udqs  -512(%edx){1to16}, %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7c,0xdf,0x6c,0x52,0x80]
          vcvttps2udqs  -512(%edx){1to16}, %zmm2 {%k7} {z}

// CHECK: vcvttps2uqqs %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf5,0x7d,0x08,0x6c,0xd3]
          vcvttps2uqqs %xmm3, %xmm2

// CHECK: vcvttps2uqqs %xmm3, %xmm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7d,0x0f,0x6c,0xd3]
          vcvttps2uqqs %xmm3, %xmm2 {%k7}

// CHECK: vcvttps2uqqs %xmm3, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7d,0x8f,0x6c,0xd3]
          vcvttps2uqqs %xmm3, %xmm2 {%k7} {z}

// CHECK: vcvttps2uqqs %xmm3, %ymm2
// CHECK: encoding: [0x62,0xf5,0x7d,0x28,0x6c,0xd3]
          vcvttps2uqqs %xmm3, %ymm2

// CHECK: vcvttps2uqqs {sae}, %xmm3, %ymm2
// CHECK: encoding: [0x62,0xf5,0x79,0x18,0x6c,0xd3]
          vcvttps2uqqs {sae}, %xmm3, %ymm2

// CHECK: vcvttps2uqqs %xmm3, %ymm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7d,0x2f,0x6c,0xd3]
          vcvttps2uqqs %xmm3, %ymm2 {%k7}

// CHECK: vcvttps2uqqs {sae}, %xmm3, %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x79,0x9f,0x6c,0xd3]
          vcvttps2uqqs {sae}, %xmm3, %ymm2 {%k7} {z}

// CHECK: vcvttps2uqqs %ymm3, %zmm2
// CHECK: encoding: [0x62,0xf5,0x7d,0x48,0x6c,0xd3]
          vcvttps2uqqs %ymm3, %zmm2

// CHECK: vcvttps2uqqs {sae}, %ymm3, %zmm2
// CHECK: encoding: [0x62,0xf5,0x7d,0x18,0x6c,0xd3]
          vcvttps2uqqs {sae}, %ymm3, %zmm2

// CHECK: vcvttps2uqqs %ymm3, %zmm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7d,0x4f,0x6c,0xd3]
          vcvttps2uqqs %ymm3, %zmm2 {%k7}

// CHECK: vcvttps2uqqs {sae}, %ymm3, %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7d,0x9f,0x6c,0xd3]
          vcvttps2uqqs {sae}, %ymm3, %zmm2 {%k7} {z}

// CHECK: vcvttps2uqqs  268435456(%esp,%esi,8), %xmm2
// CHECK: encoding: [0x62,0xf5,0x7d,0x08,0x6c,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvttps2uqqs  268435456(%esp,%esi,8), %xmm2

// CHECK: vcvttps2uqqs  291(%edi,%eax,4), %xmm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7d,0x0f,0x6c,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvttps2uqqs  291(%edi,%eax,4), %xmm2 {%k7}

// CHECK: vcvttps2uqqs  (%eax){1to2}, %xmm2
// CHECK: encoding: [0x62,0xf5,0x7d,0x18,0x6c,0x10]
          vcvttps2uqqs  (%eax){1to2}, %xmm2

// CHECK: vcvttps2uqqs  -256(,%ebp,2), %xmm2
// CHECK: encoding: [0x62,0xf5,0x7d,0x08,0x6c,0x14,0x6d,0x00,0xff,0xff,0xff]
          vcvttps2uqqs  -256(,%ebp,2), %xmm2

// CHECK: vcvttps2uqqs  1016(%ecx), %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7d,0x8f,0x6c,0x51,0x7f]
          vcvttps2uqqs  1016(%ecx), %xmm2 {%k7} {z}

// CHECK: vcvttps2uqqs  -512(%edx){1to2}, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7d,0x9f,0x6c,0x52,0x80]
          vcvttps2uqqs  -512(%edx){1to2}, %xmm2 {%k7} {z}

// CHECK: vcvttps2uqqs  268435456(%esp,%esi,8), %ymm2
// CHECK: encoding: [0x62,0xf5,0x7d,0x28,0x6c,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvttps2uqqs  268435456(%esp,%esi,8), %ymm2

// CHECK: vcvttps2uqqs  291(%edi,%eax,4), %ymm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7d,0x2f,0x6c,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvttps2uqqs  291(%edi,%eax,4), %ymm2 {%k7}

// CHECK: vcvttps2uqqs  (%eax){1to4}, %ymm2
// CHECK: encoding: [0x62,0xf5,0x7d,0x38,0x6c,0x10]
          vcvttps2uqqs  (%eax){1to4}, %ymm2

// CHECK: vcvttps2uqqs  -512(,%ebp,2), %ymm2
// CHECK: encoding: [0x62,0xf5,0x7d,0x28,0x6c,0x14,0x6d,0x00,0xfe,0xff,0xff]
          vcvttps2uqqs  -512(,%ebp,2), %ymm2

// CHECK: vcvttps2uqqs  2032(%ecx), %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7d,0xaf,0x6c,0x51,0x7f]
          vcvttps2uqqs  2032(%ecx), %ymm2 {%k7} {z}

// CHECK: vcvttps2uqqs  -512(%edx){1to4}, %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7d,0xbf,0x6c,0x52,0x80]
          vcvttps2uqqs  -512(%edx){1to4}, %ymm2 {%k7} {z}

// CHECK: vcvttps2uqqs  268435456(%esp,%esi,8), %zmm2
// CHECK: encoding: [0x62,0xf5,0x7d,0x48,0x6c,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvttps2uqqs  268435456(%esp,%esi,8), %zmm2

// CHECK: vcvttps2uqqs  291(%edi,%eax,4), %zmm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7d,0x4f,0x6c,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvttps2uqqs  291(%edi,%eax,4), %zmm2 {%k7}

// CHECK: vcvttps2uqqs  (%eax){1to8}, %zmm2
// CHECK: encoding: [0x62,0xf5,0x7d,0x58,0x6c,0x10]
          vcvttps2uqqs  (%eax){1to8}, %zmm2

// CHECK: vcvttps2uqqs  -1024(,%ebp,2), %zmm2
// CHECK: encoding: [0x62,0xf5,0x7d,0x48,0x6c,0x14,0x6d,0x00,0xfc,0xff,0xff]
          vcvttps2uqqs  -1024(,%ebp,2), %zmm2

// CHECK: vcvttps2uqqs  4064(%ecx), %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7d,0xcf,0x6c,0x51,0x7f]
          vcvttps2uqqs  4064(%ecx), %zmm2 {%k7} {z}

// CHECK: vcvttps2uqqs  -512(%edx){1to8}, %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7d,0xdf,0x6c,0x52,0x80]
          vcvttps2uqqs  -512(%edx){1to8}, %zmm2 {%k7} {z}

