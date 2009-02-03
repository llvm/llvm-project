// RUN: %llvmgcc %s -S -o -
// PR 1861

typedef unsigned char __u8;
typedef unsigned int __u32;
typedef unsigned short u16;
typedef __u32 __le32;
struct bcm43xx_plcp_hdr6 {
  union {
    __le32 data;
    __u8 raw[6];
  }
    __attribute__((__packed__));
}
  __attribute__((__packed__));
struct bcm43xx_txhdr {
  union {
    struct {
      struct bcm43xx_plcp_hdr6 plcp;
    };
  };
}
  __attribute__((__packed__));
static void bcm43xx_generate_rts(struct bcm43xx_txhdr *txhdr ) { }
