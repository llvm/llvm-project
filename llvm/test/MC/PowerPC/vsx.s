# RUN: llvm-mc -triple powerpc64-unknown-linux-gnu --show-encoding %s | FileCheck -check-prefix=CHECK-BE %s
# RUN: llvm-mc -triple powerpc64le-unknown-linux-gnu --show-encoding %s | FileCheck -check-prefix=CHECK-LE %s

# CHECK-BE: xxswapd 7, 63                      # encoding: [0xf0,0xff,0xfa,0x56]
# CHECK-LE: xxswapd 7, 63                      # encoding: [0x56,0xfa,0xff,0xf0]
            xxswapd %vs7, %vs63

# CHECK-BE: lxsdx 39, 5, 31                    # encoding: [0x7c,0xe5,0xfc,0x99]
# CHECK-LE: lxsdx 39, 5, 31                    # encoding: [0x99,0xfc,0xe5,0x7c]
            lxsdx 39, 5, 31
# CHECK-BE: lxsiwax 39, 5, 31                  # encoding: [0x7c,0xe5,0xf8,0x99]
# CHECK-LE: lxsiwax 39, 5, 31                  # encoding: [0x99,0xf8,0xe5,0x7c]
            lxsiwax 39, 5, 31
# CHECK-BE: lxsiwzx 39, 5, 31                  # encoding: [0x7c,0xe5,0xf8,0x19]
# CHECK-LE: lxsiwzx 39, 5, 31                  # encoding: [0x19,0xf8,0xe5,0x7c]
            lxsiwzx 39, 5, 31
# CHECK-BE: lxsspx 39, 5, 31                   # encoding: [0x7c,0xe5,0xfc,0x19]
# CHECK-LE: lxsspx 39, 5, 31                   # encoding: [0x19,0xfc,0xe5,0x7c]
            lxsspx 39, 5, 31
# CHECK-BE: lxvd2x 39, 5, 31                   # encoding: [0x7c,0xe5,0xfe,0x99]
# CHECK-LE: lxvd2x 39, 5, 31                   # encoding: [0x99,0xfe,0xe5,0x7c]
            lxvd2x 39, 5, 31
# CHECK-BE: lxvdsx 39, 5, 31                   # encoding: [0x7c,0xe5,0xfa,0x99]
# CHECK-LE: lxvdsx 39, 5, 31                   # encoding: [0x99,0xfa,0xe5,0x7c]
            lxvdsx 39, 5, 31
# CHECK-BE: lxvw4x 39, 5, 31                   # encoding: [0x7c,0xe5,0xfe,0x19]
# CHECK-LE: lxvw4x 39, 5, 31                   # encoding: [0x19,0xfe,0xe5,0x7c]
            lxvw4x 39, 5, 31
# CHECK-BE: stxsdx 40, 5, 31                   # encoding: [0x7d,0x05,0xfd,0x99]
# CHECK-LE: stxsdx 40, 5, 31                   # encoding: [0x99,0xfd,0x05,0x7d]
            stxsdx 40, 5, 31
# CHECK-BE: stxsiwx 40, 5, 31                  # encoding: [0x7d,0x05,0xf9,0x19]
# CHECK-LE: stxsiwx 40, 5, 31                  # encoding: [0x19,0xf9,0x05,0x7d]
            stxsiwx 40, 5, 31
# CHECK-BE: stxsspx 40, 5, 31                  # encoding: [0x7d,0x05,0xfd,0x19]
# CHECK-LE: stxsspx 40, 5, 31                  # encoding: [0x19,0xfd,0x05,0x7d]
            stxsspx 40, 5, 31
# CHECK-BE: stxvd2x 40, 5, 31                  # encoding: [0x7d,0x05,0xff,0x99]
# CHECK-LE: stxvd2x 40, 5, 31                  # encoding: [0x99,0xff,0x05,0x7d]
            stxvd2x 40, 5, 31
# CHECK-BE: stxvw4x 40, 5, 31                  # encoding: [0x7d,0x05,0xff,0x19]
# CHECK-LE: stxvw4x 40, 5, 31                  # encoding: [0x19,0xff,0x05,0x7d]
            stxvw4x 40, 5, 31
# CHECK-BE: xsabsdp 7, 27                      # encoding: [0xf0,0xe0,0xdd,0x64]
# CHECK-LE: xsabsdp 7, 27                      # encoding: [0x64,0xdd,0xe0,0xf0]
            xsabsdp 7, 27
# CHECK-BE: xsaddsp 7, 63, 27                  # encoding: [0xf0,0xff,0xd8,0x04]
# CHECK-LE: xsaddsp 7, 63, 27                  # encoding: [0x04,0xd8,0xff,0xf0]
            xsaddsp 7, 63, 27
# CHECK-BE: xsadddp 7, 63, 27                  # encoding: [0xf0,0xff,0xd9,0x04]
# CHECK-LE: xsadddp 7, 63, 27                  # encoding: [0x04,0xd9,0xff,0xf0]
            xsadddp 7, 63, 27
# CHECK-BE: xscmpodp 6, 63, 27                 # encoding: [0xf3,0x1f,0xd9,0x5c]
# CHECK-LE: xscmpodp 6, 63, 27                 # encoding: [0x5c,0xd9,0x1f,0xf3]
            xscmpodp 6, 63, 27
# CHECK-BE: xscmpudp 6, 63, 27                 # encoding: [0xf3,0x1f,0xd9,0x1c]
# CHECK-LE: xscmpudp 6, 63, 27                 # encoding: [0x1c,0xd9,0x1f,0xf3]
            xscmpudp 6, 63, 27
# CHECK-BE: xscpsgndp 7, 63, 27                # encoding: [0xf0,0xff,0xdd,0x84]
# CHECK-LE: xscpsgndp 7, 63, 27                # encoding: [0x84,0xdd,0xff,0xf0]
            xscpsgndp 7, 63, 27
# CHECK-BE: xscvdpsp 7, 27                     # encoding: [0xf0,0xe0,0xdc,0x24]
# CHECK-LE: xscvdpsp 7, 27                     # encoding: [0x24,0xdc,0xe0,0xf0]
            xscvdpsp 7, 27
# CHECK-BE: xscvdpspn 7, 27                    # encoding: [0xf0,0xe0,0xdc,0x2c]
# CHECK-LE: xscvdpspn 7, 27                    # encoding: [0x2c,0xdc,0xe0,0xf0]
            xscvdpspn 7, 27
# CHECK-BE: xscvdpsxds 7, 27                   # encoding: [0xf0,0xe0,0xdd,0x60]
# CHECK-LE: xscvdpsxds 7, 27                   # encoding: [0x60,0xdd,0xe0,0xf0]
            xscvdpsxds 7, 27
# CHECK-BE: xscvdpsxws 7, 27                   # encoding: [0xf0,0xe0,0xd9,0x60]
# CHECK-LE: xscvdpsxws 7, 27                   # encoding: [0x60,0xd9,0xe0,0xf0]
            xscvdpsxws 7, 27
# CHECK-BE: xscvdpuxds 7, 27                   # encoding: [0xf0,0xe0,0xdd,0x20]
# CHECK-LE: xscvdpuxds 7, 27                   # encoding: [0x20,0xdd,0xe0,0xf0]
            xscvdpuxds 7, 27
# CHECK-BE: xscvdpuxws 7, 27                   # encoding: [0xf0,0xe0,0xd9,0x20]
# CHECK-LE: xscvdpuxws 7, 27                   # encoding: [0x20,0xd9,0xe0,0xf0]
            xscvdpuxws 7, 27
# CHECK-BE: xscvspdp 7, 27                     # encoding: [0xf0,0xe0,0xdd,0x24]
# CHECK-LE: xscvspdp 7, 27                     # encoding: [0x24,0xdd,0xe0,0xf0]
            xscvspdp 7, 27
# CHECK-BE: xscvspdpn 7, 27                    # encoding: [0xf0,0xe0,0xdd,0x2c]
# CHECK-LE: xscvspdpn 7, 27                    # encoding: [0x2c,0xdd,0xe0,0xf0]
            xscvspdpn 7, 27
# CHECK-BE: xscvsxdsp 7, 27                    # encoding: [0xf0,0xe0,0xdc,0xe0]
# CHECK-LE: xscvsxdsp 7, 27                    # encoding: [0xe0,0xdc,0xe0,0xf0]
            xscvsxdsp 7, 27
# CHECK-BE: xscvsxddp 7, 27                    # encoding: [0xf0,0xe0,0xdd,0xe0]
# CHECK-LE: xscvsxddp 7, 27                    # encoding: [0xe0,0xdd,0xe0,0xf0]
            xscvsxddp 7, 27
# CHECK-BE: xscvuxdsp 7, 27                    # encoding: [0xf0,0xe0,0xdc,0xa0]
# CHECK-LE: xscvuxdsp 7, 27                    # encoding: [0xa0,0xdc,0xe0,0xf0]
            xscvuxdsp 7, 27
# CHECK-BE: xscvuxddp 7, 27                    # encoding: [0xf0,0xe0,0xdd,0xa0]
# CHECK-LE: xscvuxddp 7, 27                    # encoding: [0xa0,0xdd,0xe0,0xf0]
            xscvuxddp 7, 27
# CHECK-BE: xsdivsp 7, 63, 27                  # encoding: [0xf0,0xff,0xd8,0xc4]
# CHECK-LE: xsdivsp 7, 63, 27                  # encoding: [0xc4,0xd8,0xff,0xf0]
            xsdivsp 7, 63, 27
# CHECK-BE: xsdivdp 7, 63, 27                  # encoding: [0xf0,0xff,0xd9,0xc4]
# CHECK-LE: xsdivdp 7, 63, 27                  # encoding: [0xc4,0xd9,0xff,0xf0]
            xsdivdp 7, 63, 27
# CHECK-BE: xsmaddadp 7, 63, 27                # encoding: [0xf0,0xff,0xd9,0x0c]
# CHECK-LE: xsmaddadp 7, 63, 27                # encoding: [0x0c,0xd9,0xff,0xf0]
            xsmaddadp 7, 63, 27
# CHECK-BE: xsmaddmdp 7, 63, 27                # encoding: [0xf0,0xff,0xd9,0x4c]
# CHECK-LE: xsmaddmdp 7, 63, 27                # encoding: [0x4c,0xd9,0xff,0xf0]
            xsmaddmdp 7, 63, 27
# CHECK-BE: xsmaddasp 7, 63, 27                # encoding: [0xf0,0xff,0xd8,0x0c]
# CHECK-LE: xsmaddasp 7, 63, 27                # encoding: [0x0c,0xd8,0xff,0xf0]
            xsmaddasp 7, 63, 27
# CHECK-BE: xsmaddmsp 7, 63, 27                # encoding: [0xf0,0xff,0xd8,0x4c]
# CHECK-LE: xsmaddmsp 7, 63, 27                # encoding: [0x4c,0xd8,0xff,0xf0]
            xsmaddmsp 7, 63, 27
# CHECK-BE: xsmaxdp 7, 63, 27                  # encoding: [0xf0,0xff,0xdd,0x04]
# CHECK-LE: xsmaxdp 7, 63, 27                  # encoding: [0x04,0xdd,0xff,0xf0]
            xsmaxdp 7, 63, 27
# CHECK-BE: xsmindp 7, 63, 27                  # encoding: [0xf0,0xff,0xdd,0x44]
# CHECK-LE: xsmindp 7, 63, 27                  # encoding: [0x44,0xdd,0xff,0xf0]
            xsmindp 7, 63, 27
# CHECK-BE: xsmsubadp 7, 63, 27                # encoding: [0xf0,0xff,0xd9,0x8c]
# CHECK-LE: xsmsubadp 7, 63, 27                # encoding: [0x8c,0xd9,0xff,0xf0]
            xsmsubadp 7, 63, 27
# CHECK-BE: xsmsubmdp 7, 63, 27                # encoding: [0xf0,0xff,0xd9,0xcc]
# CHECK-LE: xsmsubmdp 7, 63, 27                # encoding: [0xcc,0xd9,0xff,0xf0]
            xsmsubmdp 7, 63, 27
# CHECK-BE: xsmsubasp 7, 63, 27                # encoding: [0xf0,0xff,0xd8,0x8c]
# CHECK-LE: xsmsubasp 7, 63, 27                # encoding: [0x8c,0xd8,0xff,0xf0]
            xsmsubasp 7, 63, 27
# CHECK-BE: xsmsubmsp 7, 63, 27                # encoding: [0xf0,0xff,0xd8,0xcc]
# CHECK-LE: xsmsubmsp 7, 63, 27                # encoding: [0xcc,0xd8,0xff,0xf0]
            xsmsubmsp 7, 63, 27
# CHECK-BE: xsmulsp 7, 63, 27                  # encoding: [0xf0,0xff,0xd8,0x84]
# CHECK-LE: xsmulsp 7, 63, 27                  # encoding: [0x84,0xd8,0xff,0xf0]
            xsmulsp 7, 63, 27
# CHECK-BE: xsmuldp 7, 63, 27                  # encoding: [0xf0,0xff,0xd9,0x84]
# CHECK-LE: xsmuldp 7, 63, 27                  # encoding: [0x84,0xd9,0xff,0xf0]
            xsmuldp 7, 63, 27
# CHECK-BE: xsnabsdp 7, 27                     # encoding: [0xf0,0xe0,0xdd,0xa4]
# CHECK-LE: xsnabsdp 7, 27                     # encoding: [0xa4,0xdd,0xe0,0xf0]
            xsnabsdp 7, 27
# CHECK-BE: xsnegdp 7, 27                      # encoding: [0xf0,0xe0,0xdd,0xe4]
# CHECK-LE: xsnegdp 7, 27                      # encoding: [0xe4,0xdd,0xe0,0xf0]
            xsnegdp 7, 27
# CHECK-BE: xsnmaddadp 7, 63, 27               # encoding: [0xf0,0xff,0xdd,0x0c]
# CHECK-LE: xsnmaddadp 7, 63, 27               # encoding: [0x0c,0xdd,0xff,0xf0]
            xsnmaddadp 7, 63, 27
# CHECK-BE: xsnmaddmdp 7, 63, 27               # encoding: [0xf0,0xff,0xdd,0x4c]
# CHECK-LE: xsnmaddmdp 7, 63, 27               # encoding: [0x4c,0xdd,0xff,0xf0]
            xsnmaddmdp 7, 63, 27
# CHECK-BE: xsnmsubadp 7, 63, 27               # encoding: [0xf0,0xff,0xdd,0x8c]
# CHECK-LE: xsnmsubadp 7, 63, 27               # encoding: [0x8c,0xdd,0xff,0xf0]
            xsnmsubadp 7, 63, 27
# CHECK-BE: xsnmsubmdp 7, 63, 27               # encoding: [0xf0,0xff,0xdd,0xcc]
# CHECK-LE: xsnmsubmdp 7, 63, 27               # encoding: [0xcc,0xdd,0xff,0xf0]
            xsnmsubmdp 7, 63, 27
# CHECK-BE: xsnmaddasp 7, 63, 27               # encoding: [0xf0,0xff,0xdc,0x0c]
# CHECK-LE: xsnmaddasp 7, 63, 27               # encoding: [0x0c,0xdc,0xff,0xf0]
            xsnmaddasp 7, 63, 27
# CHECK-BE: xsnmaddmsp 7, 63, 27               # encoding: [0xf0,0xff,0xdc,0x4c]
# CHECK-LE: xsnmaddmsp 7, 63, 27               # encoding: [0x4c,0xdc,0xff,0xf0]
            xsnmaddmsp 7, 63, 27
# CHECK-BE: xsnmsubasp 7, 63, 27               # encoding: [0xf0,0xff,0xdc,0x8c]
# CHECK-LE: xsnmsubasp 7, 63, 27               # encoding: [0x8c,0xdc,0xff,0xf0]
            xsnmsubasp 7, 63, 27
# CHECK-BE: xsnmsubmsp 7, 63, 27               # encoding: [0xf0,0xff,0xdc,0xcc]
# CHECK-LE: xsnmsubmsp 7, 63, 27               # encoding: [0xcc,0xdc,0xff,0xf0]
            xsnmsubmsp 7, 63, 27
# CHECK-BE: xsrdpi 7, 27                       # encoding: [0xf0,0xe0,0xd9,0x24]
# CHECK-LE: xsrdpi 7, 27                       # encoding: [0x24,0xd9,0xe0,0xf0]
            xsrdpi 7, 27
# CHECK-BE: xsrdpic 7, 27                      # encoding: [0xf0,0xe0,0xd9,0xac]
# CHECK-LE: xsrdpic 7, 27                      # encoding: [0xac,0xd9,0xe0,0xf0]
            xsrdpic 7, 27
# CHECK-BE: xsrdpim 7, 27                      # encoding: [0xf0,0xe0,0xd9,0xe4]
# CHECK-LE: xsrdpim 7, 27                      # encoding: [0xe4,0xd9,0xe0,0xf0]
            xsrdpim 7, 27
# CHECK-BE: xsrdpip 7, 27                      # encoding: [0xf0,0xe0,0xd9,0xa4]
# CHECK-LE: xsrdpip 7, 27                      # encoding: [0xa4,0xd9,0xe0,0xf0]
            xsrdpip 7, 27
# CHECK-BE: xsrdpiz 7, 27                      # encoding: [0xf0,0xe0,0xd9,0x64]
# CHECK-LE: xsrdpiz 7, 27                      # encoding: [0x64,0xd9,0xe0,0xf0]
            xsrdpiz 7, 27
# CHECK-BE: xsresp 7, 27                       # encoding: [0xf0,0xe0,0xd8,0x68]
# CHECK-LE: xsresp 7, 27                       # encoding: [0x68,0xd8,0xe0,0xf0]
            xsresp 7, 27
# CHECK-BE: xsredp 7, 27                       # encoding: [0xf0,0xe0,0xd9,0x68]
# CHECK-LE: xsredp 7, 27                       # encoding: [0x68,0xd9,0xe0,0xf0]
            xsredp 7, 27
# CHECK-BE: xsrsqrtesp 7, 27                   # encoding: [0xf0,0xe0,0xd8,0x28]
# CHECK-LE: xsrsqrtesp 7, 27                   # encoding: [0x28,0xd8,0xe0,0xf0]
            xsrsqrtesp 7, 27
# CHECK-BE: xsrsqrtedp 7, 27                   # encoding: [0xf0,0xe0,0xd9,0x28]
# CHECK-LE: xsrsqrtedp 7, 27                   # encoding: [0x28,0xd9,0xe0,0xf0]
            xsrsqrtedp 7, 27
# CHECK-BE: xssqrtsp 7, 27                     # encoding: [0xf0,0xe0,0xd8,0x2c]
# CHECK-LE: xssqrtsp 7, 27                     # encoding: [0x2c,0xd8,0xe0,0xf0]
            xssqrtsp 7, 27
# CHECK-BE: xssqrtdp 7, 27                     # encoding: [0xf0,0xe0,0xd9,0x2c]
# CHECK-LE: xssqrtdp 7, 27                     # encoding: [0x2c,0xd9,0xe0,0xf0]
            xssqrtdp 7, 27
# CHECK-BE: xssubsp 7, 63, 27                  # encoding: [0xf0,0xff,0xd8,0x44]
# CHECK-LE: xssubsp 7, 63, 27                  # encoding: [0x44,0xd8,0xff,0xf0]
            xssubsp 7, 63, 27
# CHECK-BE: xssubdp 7, 63, 27                  # encoding: [0xf0,0xff,0xd9,0x44]
# CHECK-LE: xssubdp 7, 63, 27                  # encoding: [0x44,0xd9,0xff,0xf0]
            xssubdp 7, 63, 27
# CHECK-BE: xstdivdp 6, 63, 27                 # encoding: [0xf3,0x1f,0xd9,0xec]
# CHECK-LE: xstdivdp 6, 63, 27                 # encoding: [0xec,0xd9,0x1f,0xf3]
            xstdivdp 6, 63, 27
# CHECK-BE: xstsqrtdp 6, 27                    # encoding: [0xf3,0x00,0xd9,0xa8]
# CHECK-LE: xstsqrtdp 6, 27                    # encoding: [0xa8,0xd9,0x00,0xf3]
            xstsqrtdp 6, 27
# CHECK-BE: xvabsdp 7, 27                      # encoding: [0xf0,0xe0,0xdf,0x64]
# CHECK-LE: xvabsdp 7, 27                      # encoding: [0x64,0xdf,0xe0,0xf0]
            xvabsdp 7, 27
# CHECK-BE: xvabssp 7, 27                      # encoding: [0xf0,0xe0,0xde,0x64]
# CHECK-LE: xvabssp 7, 27                      # encoding: [0x64,0xde,0xe0,0xf0]
            xvabssp 7, 27
# CHECK-BE: xvadddp 7, 63, 27                  # encoding: [0xf0,0xff,0xdb,0x04]
# CHECK-LE: xvadddp 7, 63, 27                  # encoding: [0x04,0xdb,0xff,0xf0]
            xvadddp 7, 63, 27
# CHECK-BE: xvaddsp 7, 63, 27                  # encoding: [0xf0,0xff,0xda,0x04]
# CHECK-LE: xvaddsp 7, 63, 27                  # encoding: [0x04,0xda,0xff,0xf0]
            xvaddsp 7, 63, 27
# CHECK-BE: xvcmpeqdp 7, 63, 27                # encoding: [0xf0,0xff,0xdb,0x1c]
# CHECK-LE: xvcmpeqdp 7, 63, 27                # encoding: [0x1c,0xdb,0xff,0xf0]
            xvcmpeqdp 7, 63, 27
# CHECK-BE: xvcmpeqdp. 7, 63, 27               # encoding: [0xf0,0xff,0xdf,0x1c]
# CHECK-LE: xvcmpeqdp. 7, 63, 27               # encoding: [0x1c,0xdf,0xff,0xf0]
            xvcmpeqdp. 7, 63, 27
# CHECK-BE: xvcmpeqsp 7, 63, 27                # encoding: [0xf0,0xff,0xda,0x1c]
# CHECK-LE: xvcmpeqsp 7, 63, 27                # encoding: [0x1c,0xda,0xff,0xf0]
            xvcmpeqsp 7, 63, 27
# CHECK-BE: xvcmpeqsp. 7, 63, 27               # encoding: [0xf0,0xff,0xde,0x1c]
# CHECK-LE: xvcmpeqsp. 7, 63, 27               # encoding: [0x1c,0xde,0xff,0xf0]
            xvcmpeqsp. 7, 63, 27
# CHECK-BE: xvcmpgedp 7, 63, 27                # encoding: [0xf0,0xff,0xdb,0x9c]
# CHECK-LE: xvcmpgedp 7, 63, 27                # encoding: [0x9c,0xdb,0xff,0xf0]
            xvcmpgedp 7, 63, 27
# CHECK-BE: xvcmpgedp. 7, 63, 27               # encoding: [0xf0,0xff,0xdf,0x9c]
# CHECK-LE: xvcmpgedp. 7, 63, 27               # encoding: [0x9c,0xdf,0xff,0xf0]
            xvcmpgedp. 7, 63, 27
# CHECK-BE: xvcmpgesp 7, 63, 27                # encoding: [0xf0,0xff,0xda,0x9c]
# CHECK-LE: xvcmpgesp 7, 63, 27                # encoding: [0x9c,0xda,0xff,0xf0]
            xvcmpgesp 7, 63, 27
# CHECK-BE: xvcmpgesp. 7, 63, 27               # encoding: [0xf0,0xff,0xde,0x9c]
# CHECK-LE: xvcmpgesp. 7, 63, 27               # encoding: [0x9c,0xde,0xff,0xf0]
            xvcmpgesp. 7, 63, 27
# CHECK-BE: xvcmpgtdp 7, 63, 27                # encoding: [0xf0,0xff,0xdb,0x5c]
# CHECK-LE: xvcmpgtdp 7, 63, 27                # encoding: [0x5c,0xdb,0xff,0xf0]
            xvcmpgtdp 7, 63, 27
# CHECK-BE: xvcmpgtdp. 7, 63, 27               # encoding: [0xf0,0xff,0xdf,0x5c]
# CHECK-LE: xvcmpgtdp. 7, 63, 27               # encoding: [0x5c,0xdf,0xff,0xf0]
            xvcmpgtdp. 7, 63, 27
# CHECK-BE: xvcmpgtsp 7, 63, 27                # encoding: [0xf0,0xff,0xda,0x5c]
# CHECK-LE: xvcmpgtsp 7, 63, 27                # encoding: [0x5c,0xda,0xff,0xf0]
            xvcmpgtsp 7, 63, 27
# CHECK-BE: xvcmpgtsp. 7, 63, 27               # encoding: [0xf0,0xff,0xde,0x5c]
# CHECK-LE: xvcmpgtsp. 7, 63, 27               # encoding: [0x5c,0xde,0xff,0xf0]
            xvcmpgtsp. 7, 63, 27
# CHECK-BE: xvcpsgndp 7, 63, 27                # encoding: [0xf0,0xff,0xdf,0x84]
# CHECK-LE: xvcpsgndp 7, 63, 27                # encoding: [0x84,0xdf,0xff,0xf0]
            xvcpsgndp 7, 63, 27
# CHECK-BE: xvcpsgnsp 7, 63, 27                # encoding: [0xf0,0xff,0xde,0x84]
# CHECK-LE: xvcpsgnsp 7, 63, 27                # encoding: [0x84,0xde,0xff,0xf0]
            xvcpsgnsp 7, 63, 27
# CHECK-BE: xvcvdpsp 7, 27                     # encoding: [0xf0,0xe0,0xde,0x24]
# CHECK-LE: xvcvdpsp 7, 27                     # encoding: [0x24,0xde,0xe0,0xf0]
            xvcvdpsp 7, 27
# CHECK-BE: xvcvdpsxds 7, 27                   # encoding: [0xf0,0xe0,0xdf,0x60]
# CHECK-LE: xvcvdpsxds 7, 27                   # encoding: [0x60,0xdf,0xe0,0xf0]
            xvcvdpsxds 7, 27
# CHECK-BE: xvcvdpsxws 7, 27                   # encoding: [0xf0,0xe0,0xdb,0x60]
# CHECK-LE: xvcvdpsxws 7, 27                   # encoding: [0x60,0xdb,0xe0,0xf0]
            xvcvdpsxws 7, 27
# CHECK-BE: xvcvdpuxds 7, 27                   # encoding: [0xf0,0xe0,0xdf,0x20]
# CHECK-LE: xvcvdpuxds 7, 27                   # encoding: [0x20,0xdf,0xe0,0xf0]
            xvcvdpuxds 7, 27
# CHECK-BE: xvcvdpuxws 7, 27                   # encoding: [0xf0,0xe0,0xdb,0x20]
# CHECK-LE: xvcvdpuxws 7, 27                   # encoding: [0x20,0xdb,0xe0,0xf0]
            xvcvdpuxws 7, 27
# CHECK-BE: xvcvspdp 7, 27                     # encoding: [0xf0,0xe0,0xdf,0x24]
# CHECK-LE: xvcvspdp 7, 27                     # encoding: [0x24,0xdf,0xe0,0xf0]
            xvcvspdp 7, 27
# CHECK-BE: xvcvspsxds 7, 27                   # encoding: [0xf0,0xe0,0xde,0x60]
# CHECK-LE: xvcvspsxds 7, 27                   # encoding: [0x60,0xde,0xe0,0xf0]
            xvcvspsxds 7, 27
# CHECK-BE: xvcvspsxws 7, 27                   # encoding: [0xf0,0xe0,0xda,0x60]
# CHECK-LE: xvcvspsxws 7, 27                   # encoding: [0x60,0xda,0xe0,0xf0]
            xvcvspsxws 7, 27
# CHECK-BE: xvcvspuxds 7, 27                   # encoding: [0xf0,0xe0,0xde,0x20]
# CHECK-LE: xvcvspuxds 7, 27                   # encoding: [0x20,0xde,0xe0,0xf0]
            xvcvspuxds 7, 27
# CHECK-BE: xvcvspuxws 7, 27                   # encoding: [0xf0,0xe0,0xda,0x20]
# CHECK-LE: xvcvspuxws 7, 27                   # encoding: [0x20,0xda,0xe0,0xf0]
            xvcvspuxws 7, 27
# CHECK-BE: xvcvsxddp 7, 27                    # encoding: [0xf0,0xe0,0xdf,0xe0]
# CHECK-LE: xvcvsxddp 7, 27                    # encoding: [0xe0,0xdf,0xe0,0xf0]
            xvcvsxddp 7, 27
# CHECK-BE: xvcvsxdsp 7, 27                    # encoding: [0xf0,0xe0,0xde,0xe0]
# CHECK-LE: xvcvsxdsp 7, 27                    # encoding: [0xe0,0xde,0xe0,0xf0]
            xvcvsxdsp 7, 27
# CHECK-BE: xvcvsxwdp 7, 27                    # encoding: [0xf0,0xe0,0xdb,0xe0]
# CHECK-LE: xvcvsxwdp 7, 27                    # encoding: [0xe0,0xdb,0xe0,0xf0]
            xvcvsxwdp 7, 27
# CHECK-BE: xvcvsxwsp 7, 27                    # encoding: [0xf0,0xe0,0xda,0xe0]
# CHECK-LE: xvcvsxwsp 7, 27                    # encoding: [0xe0,0xda,0xe0,0xf0]
            xvcvsxwsp 7, 27
# CHECK-BE: xvcvuxddp 7, 27                    # encoding: [0xf0,0xe0,0xdf,0xa0]
# CHECK-LE: xvcvuxddp 7, 27                    # encoding: [0xa0,0xdf,0xe0,0xf0]
            xvcvuxddp 7, 27
# CHECK-BE: xvcvuxdsp 7, 27                    # encoding: [0xf0,0xe0,0xde,0xa0]
# CHECK-LE: xvcvuxdsp 7, 27                    # encoding: [0xa0,0xde,0xe0,0xf0]
            xvcvuxdsp 7, 27
# CHECK-BE: xvcvuxwdp 7, 27                    # encoding: [0xf0,0xe0,0xdb,0xa0]
# CHECK-LE: xvcvuxwdp 7, 27                    # encoding: [0xa0,0xdb,0xe0,0xf0]
            xvcvuxwdp 7, 27
# CHECK-BE: xvcvuxwsp 7, 27                    # encoding: [0xf0,0xe0,0xda,0xa0]
# CHECK-LE: xvcvuxwsp 7, 27                    # encoding: [0xa0,0xda,0xe0,0xf0]
            xvcvuxwsp 7, 27
# CHECK-BE: xvdivdp 7, 63, 27                  # encoding: [0xf0,0xff,0xdb,0xc4]
# CHECK-LE: xvdivdp 7, 63, 27                  # encoding: [0xc4,0xdb,0xff,0xf0]
            xvdivdp 7, 63, 27
# CHECK-BE: xvdivsp 7, 63, 27                  # encoding: [0xf0,0xff,0xda,0xc4]
# CHECK-LE: xvdivsp 7, 63, 27                  # encoding: [0xc4,0xda,0xff,0xf0]
            xvdivsp 7, 63, 27
# CHECK-BE: xvmaddadp 7, 63, 27                # encoding: [0xf0,0xff,0xdb,0x0c]
# CHECK-LE: xvmaddadp 7, 63, 27                # encoding: [0x0c,0xdb,0xff,0xf0]
            xvmaddadp 7, 63, 27
# CHECK-BE: xvmaddasp 7, 63, 27                # encoding: [0xf0,0xff,0xda,0x0c]
# CHECK-LE: xvmaddasp 7, 63, 27                # encoding: [0x0c,0xda,0xff,0xf0]
            xvmaddasp 7, 63, 27
# CHECK-BE: xvmaddmdp 7, 63, 27                # encoding: [0xf0,0xff,0xdb,0x4c]
# CHECK-LE: xvmaddmdp 7, 63, 27                # encoding: [0x4c,0xdb,0xff,0xf0]
            xvmaddmdp 7, 63, 27
# CHECK-BE: xvmaddmsp 7, 63, 27                # encoding: [0xf0,0xff,0xda,0x4c]
# CHECK-LE: xvmaddmsp 7, 63, 27                # encoding: [0x4c,0xda,0xff,0xf0]
            xvmaddmsp 7, 63, 27
# CHECK-BE: xvmaxdp 7, 63, 27                  # encoding: [0xf0,0xff,0xdf,0x04]
# CHECK-LE: xvmaxdp 7, 63, 27                  # encoding: [0x04,0xdf,0xff,0xf0]
            xvmaxdp 7, 63, 27
# CHECK-BE: xvmaxsp 7, 63, 27                  # encoding: [0xf0,0xff,0xde,0x04]
# CHECK-LE: xvmaxsp 7, 63, 27                  # encoding: [0x04,0xde,0xff,0xf0]
            xvmaxsp 7, 63, 27
# CHECK-BE: xvmindp 7, 63, 27                  # encoding: [0xf0,0xff,0xdf,0x44]
# CHECK-LE: xvmindp 7, 63, 27                  # encoding: [0x44,0xdf,0xff,0xf0]
            xvmindp 7, 63, 27
# CHECK-BE: xvminsp 7, 63, 27                  # encoding: [0xf0,0xff,0xde,0x44]
# CHECK-LE: xvminsp 7, 63, 27                  # encoding: [0x44,0xde,0xff,0xf0]
            xvminsp 7, 63, 27
# CHECK-BE: xvmovdp 7, 63                      # encoding: [0xf0,0xff,0xff,0x86]
# CHECK-LE: xvmovdp 7, 63                      # encoding: [0x86,0xff,0xff,0xf0]
            xvmovdp 7, 63
# CHECK-BE: xvmovsp 7, 63                      # encoding: [0xf0,0xff,0xfe,0x86]
# CHECK-LE: xvmovsp 7, 63                      # encoding: [0x86,0xfe,0xff,0xf0]
            xvmovsp 7, 63
# CHECK-BE: xvmsubadp 7, 63, 27                # encoding: [0xf0,0xff,0xdb,0x8c]
# CHECK-LE: xvmsubadp 7, 63, 27                # encoding: [0x8c,0xdb,0xff,0xf0]
            xvmsubadp 7, 63, 27
# CHECK-BE: xvmsubasp 7, 63, 27                # encoding: [0xf0,0xff,0xda,0x8c]
# CHECK-LE: xvmsubasp 7, 63, 27                # encoding: [0x8c,0xda,0xff,0xf0]
            xvmsubasp 7, 63, 27
# CHECK-BE: xvmsubmdp 7, 63, 27                # encoding: [0xf0,0xff,0xdb,0xcc]
# CHECK-LE: xvmsubmdp 7, 63, 27                # encoding: [0xcc,0xdb,0xff,0xf0]
            xvmsubmdp 7, 63, 27
# CHECK-BE: xvmsubmsp 7, 63, 27                # encoding: [0xf0,0xff,0xda,0xcc]
# CHECK-LE: xvmsubmsp 7, 63, 27                # encoding: [0xcc,0xda,0xff,0xf0]
            xvmsubmsp 7, 63, 27
# CHECK-BE: xvmuldp 7, 63, 27                  # encoding: [0xf0,0xff,0xdb,0x84]
# CHECK-LE: xvmuldp 7, 63, 27                  # encoding: [0x84,0xdb,0xff,0xf0]
            xvmuldp 7, 63, 27
# CHECK-BE: xvmulsp 7, 63, 27                  # encoding: [0xf0,0xff,0xda,0x84]
# CHECK-LE: xvmulsp 7, 63, 27                  # encoding: [0x84,0xda,0xff,0xf0]
            xvmulsp 7, 63, 27
# CHECK-BE: xvnabsdp 7, 27                     # encoding: [0xf0,0xe0,0xdf,0xa4]
# CHECK-LE: xvnabsdp 7, 27                     # encoding: [0xa4,0xdf,0xe0,0xf0]
            xvnabsdp 7, 27
# CHECK-BE: xvnabssp 7, 27                     # encoding: [0xf0,0xe0,0xde,0xa4]
# CHECK-LE: xvnabssp 7, 27                     # encoding: [0xa4,0xde,0xe0,0xf0]
            xvnabssp 7, 27
# CHECK-BE: xvnegdp 7, 27                      # encoding: [0xf0,0xe0,0xdf,0xe4]
# CHECK-LE: xvnegdp 7, 27                      # encoding: [0xe4,0xdf,0xe0,0xf0]
            xvnegdp 7, 27
# CHECK-BE: xvnegsp 7, 27                      # encoding: [0xf0,0xe0,0xde,0xe4]
# CHECK-LE: xvnegsp 7, 27                      # encoding: [0xe4,0xde,0xe0,0xf0]
            xvnegsp 7, 27
# CHECK-BE: xvnmaddadp 7, 63, 27               # encoding: [0xf0,0xff,0xdf,0x0c]
# CHECK-LE: xvnmaddadp 7, 63, 27               # encoding: [0x0c,0xdf,0xff,0xf0]
            xvnmaddadp 7, 63, 27
# CHECK-BE: xvnmaddasp 7, 63, 27               # encoding: [0xf0,0xff,0xde,0x0c]
# CHECK-LE: xvnmaddasp 7, 63, 27               # encoding: [0x0c,0xde,0xff,0xf0]
            xvnmaddasp 7, 63, 27
# CHECK-BE: xvnmaddmdp 7, 63, 27               # encoding: [0xf0,0xff,0xdf,0x4c]
# CHECK-LE: xvnmaddmdp 7, 63, 27               # encoding: [0x4c,0xdf,0xff,0xf0]
            xvnmaddmdp 7, 63, 27
# CHECK-BE: xvnmaddmsp 7, 63, 27               # encoding: [0xf0,0xff,0xde,0x4c]
# CHECK-LE: xvnmaddmsp 7, 63, 27               # encoding: [0x4c,0xde,0xff,0xf0]
            xvnmaddmsp 7, 63, 27
# CHECK-BE: xvnmsubadp 7, 63, 27               # encoding: [0xf0,0xff,0xdf,0x8c]
# CHECK-LE: xvnmsubadp 7, 63, 27               # encoding: [0x8c,0xdf,0xff,0xf0]
            xvnmsubadp 7, 63, 27
# CHECK-BE: xvnmsubasp 7, 63, 27               # encoding: [0xf0,0xff,0xde,0x8c]
# CHECK-LE: xvnmsubasp 7, 63, 27               # encoding: [0x8c,0xde,0xff,0xf0]
            xvnmsubasp 7, 63, 27
# CHECK-BE: xvnmsubmdp 7, 63, 27               # encoding: [0xf0,0xff,0xdf,0xcc]
# CHECK-LE: xvnmsubmdp 7, 63, 27               # encoding: [0xcc,0xdf,0xff,0xf0]
            xvnmsubmdp 7, 63, 27
# CHECK-BE: xvnmsubmsp 7, 63, 27               # encoding: [0xf0,0xff,0xde,0xcc]
# CHECK-LE: xvnmsubmsp 7, 63, 27               # encoding: [0xcc,0xde,0xff,0xf0]
            xvnmsubmsp 7, 63, 27
# CHECK-BE: xvrdpi 7, 27                       # encoding: [0xf0,0xe0,0xdb,0x24]
# CHECK-LE: xvrdpi 7, 27                       # encoding: [0x24,0xdb,0xe0,0xf0]
            xvrdpi 7, 27
# CHECK-BE: xvrdpic 7, 27                      # encoding: [0xf0,0xe0,0xdb,0xac]
# CHECK-LE: xvrdpic 7, 27                      # encoding: [0xac,0xdb,0xe0,0xf0]
            xvrdpic 7, 27
# CHECK-BE: xvrdpim 7, 27                      # encoding: [0xf0,0xe0,0xdb,0xe4]
# CHECK-LE: xvrdpim 7, 27                      # encoding: [0xe4,0xdb,0xe0,0xf0]
            xvrdpim 7, 27
# CHECK-BE: xvrdpip 7, 27                      # encoding: [0xf0,0xe0,0xdb,0xa4]
# CHECK-LE: xvrdpip 7, 27                      # encoding: [0xa4,0xdb,0xe0,0xf0]
            xvrdpip 7, 27
# CHECK-BE: xvrdpiz 7, 27                      # encoding: [0xf0,0xe0,0xdb,0x64]
# CHECK-LE: xvrdpiz 7, 27                      # encoding: [0x64,0xdb,0xe0,0xf0]
            xvrdpiz 7, 27
# CHECK-BE: xvredp 7, 27                       # encoding: [0xf0,0xe0,0xdb,0x68]
# CHECK-LE: xvredp 7, 27                       # encoding: [0x68,0xdb,0xe0,0xf0]
            xvredp 7, 27
# CHECK-BE: xvresp 7, 27                       # encoding: [0xf0,0xe0,0xda,0x68]
# CHECK-LE: xvresp 7, 27                       # encoding: [0x68,0xda,0xe0,0xf0]
            xvresp 7, 27
# CHECK-BE: xvrspi 7, 27                       # encoding: [0xf0,0xe0,0xda,0x24]
# CHECK-LE: xvrspi 7, 27                       # encoding: [0x24,0xda,0xe0,0xf0]
            xvrspi 7, 27
# CHECK-BE: xvrspic 7, 27                      # encoding: [0xf0,0xe0,0xda,0xac]
# CHECK-LE: xvrspic 7, 27                      # encoding: [0xac,0xda,0xe0,0xf0]
            xvrspic 7, 27
# CHECK-BE: xvrspim 7, 27                      # encoding: [0xf0,0xe0,0xda,0xe4]
# CHECK-LE: xvrspim 7, 27                      # encoding: [0xe4,0xda,0xe0,0xf0]
            xvrspim 7, 27
# CHECK-BE: xvrspip 7, 27                      # encoding: [0xf0,0xe0,0xda,0xa4]
# CHECK-LE: xvrspip 7, 27                      # encoding: [0xa4,0xda,0xe0,0xf0]
            xvrspip 7, 27
# CHECK-BE: xvrspiz 7, 27                      # encoding: [0xf0,0xe0,0xda,0x64]
# CHECK-LE: xvrspiz 7, 27                      # encoding: [0x64,0xda,0xe0,0xf0]
            xvrspiz 7, 27
# CHECK-BE: xvrsqrtedp 7, 27                   # encoding: [0xf0,0xe0,0xdb,0x28]
# CHECK-LE: xvrsqrtedp 7, 27                   # encoding: [0x28,0xdb,0xe0,0xf0]
            xvrsqrtedp 7, 27
# CHECK-BE: xvrsqrtesp 7, 27                   # encoding: [0xf0,0xe0,0xda,0x28]
# CHECK-LE: xvrsqrtesp 7, 27                   # encoding: [0x28,0xda,0xe0,0xf0]
            xvrsqrtesp 7, 27
# CHECK-BE: xvsqrtdp 7, 27                     # encoding: [0xf0,0xe0,0xdb,0x2c]
# CHECK-LE: xvsqrtdp 7, 27                     # encoding: [0x2c,0xdb,0xe0,0xf0]
            xvsqrtdp 7, 27
# CHECK-BE: xvsqrtsp 7, 27                     # encoding: [0xf0,0xe0,0xda,0x2c]
# CHECK-LE: xvsqrtsp 7, 27                     # encoding: [0x2c,0xda,0xe0,0xf0]
            xvsqrtsp 7, 27
# CHECK-BE: xvsubdp 7, 63, 27                  # encoding: [0xf0,0xff,0xdb,0x44]
# CHECK-LE: xvsubdp 7, 63, 27                  # encoding: [0x44,0xdb,0xff,0xf0]
            xvsubdp 7, 63, 27
# CHECK-BE: xvsubsp 7, 63, 27                  # encoding: [0xf0,0xff,0xda,0x44]
# CHECK-LE: xvsubsp 7, 63, 27                  # encoding: [0x44,0xda,0xff,0xf0]
            xvsubsp 7, 63, 27
# CHECK-BE: xvtdivdp 6, 63, 27                 # encoding: [0xf3,0x1f,0xdb,0xec]
# CHECK-LE: xvtdivdp 6, 63, 27                 # encoding: [0xec,0xdb,0x1f,0xf3]
            xvtdivdp 6, 63, 27
# CHECK-BE: xvtdivsp 6, 63, 27                 # encoding: [0xf3,0x1f,0xda,0xec]
# CHECK-LE: xvtdivsp 6, 63, 27                 # encoding: [0xec,0xda,0x1f,0xf3]
            xvtdivsp 6, 63, 27
# CHECK-BE: xvtsqrtdp 6, 27                    # encoding: [0xf3,0x00,0xdb,0xa8]
# CHECK-LE: xvtsqrtdp 6, 27                    # encoding: [0xa8,0xdb,0x00,0xf3]
            xvtsqrtdp 6, 27
# CHECK-BE: xvtsqrtsp 6, 27                    # encoding: [0xf3,0x00,0xda,0xa8]
# CHECK-LE: xvtsqrtsp 6, 27                    # encoding: [0xa8,0xda,0x00,0xf3]
            xvtsqrtsp 6, 27
# CHECK-BE: xxland 7, 63, 27                   # encoding: [0xf0,0xff,0xdc,0x14]
# CHECK-LE: xxland 7, 63, 27                   # encoding: [0x14,0xdc,0xff,0xf0]
            xxland 7, 63, 27
# CHECK-BE: xxleqv 7, 63, 27                   # encoding: [0xf0,0xff,0xdd,0xd4]
# CHECK-LE: xxleqv 7, 63, 27                   # encoding: [0xd4,0xdd,0xff,0xf0]
            xxleqv 7, 63, 27
# CHECK-BE: xxlnand 7, 63, 27                  # encoding: [0xf0,0xff,0xdd,0x94]
# CHECK-LE: xxlnand 7, 63, 27                  # encoding: [0x94,0xdd,0xff,0xf0]
            xxlnand 7, 63, 27
# CHECK-BE: xxlorc 7, 63, 27                   # encoding: [0xf0,0xff,0xdd,0x54]
# CHECK-LE: xxlorc 7, 63, 27                   # encoding: [0x54,0xdd,0xff,0xf0]
            xxlorc 7, 63, 27
# CHECK-BE: xxlandc 7, 63, 27                  # encoding: [0xf0,0xff,0xdc,0x54]
# CHECK-LE: xxlandc 7, 63, 27                  # encoding: [0x54,0xdc,0xff,0xf0]
            xxlandc 7, 63, 27
# CHECK-BE: xxlnor 7, 63, 27                   # encoding: [0xf0,0xff,0xdd,0x14]
# CHECK-LE: xxlnor 7, 63, 27                   # encoding: [0x14,0xdd,0xff,0xf0]
            xxlnor 7, 63, 27
# CHECK-BE: xxlor 7, 63, 27                    # encoding: [0xf0,0xff,0xdc,0x94]
# CHECK-LE: xxlor 7, 63, 27                    # encoding: [0x94,0xdc,0xff,0xf0]
            xxlor 7, 63, 27
# CHECK-BE: xxlxor 7, 63, 27                   # encoding: [0xf0,0xff,0xdc,0xd4]
# CHECK-LE: xxlxor 7, 63, 27                   # encoding: [0xd4,0xdc,0xff,0xf0]
            xxlxor 7, 63, 27
# CHECK-BE: xxmrghd 7, 63, 27                  # encoding: [0xf0,0xff,0xd8,0x54]
# CHECK-LE: xxmrghd 7, 63, 27                  # encoding: [0x54,0xd8,0xff,0xf0]
            xxmrghd 7, 63, 27
# CHECK-BE: xxmrghw 7, 63, 27                  # encoding: [0xf0,0xff,0xd8,0x94]
# CHECK-LE: xxmrghw 7, 63, 27                  # encoding: [0x94,0xd8,0xff,0xf0]
            xxmrghw 7, 63, 27
# CHECK-BE: xxmrgld 7, 63, 27                  # encoding: [0xf0,0xff,0xdb,0x54]
# CHECK-LE: xxmrgld 7, 63, 27                  # encoding: [0x54,0xdb,0xff,0xf0]
            xxmrgld 7, 63, 27
# CHECK-BE: xxmrglw 7, 63, 27                  # encoding: [0xf0,0xff,0xd9,0x94]
# CHECK-LE: xxmrglw 7, 63, 27                  # encoding: [0x94,0xd9,0xff,0xf0]
            xxmrglw 7, 63, 27
# CHECK-BE: xxpermdi 7, 63, 27, 2              # encoding: [0xf0,0xff,0xda,0x54]
# CHECK-LE: xxpermdi 7, 63, 27, 2              # encoding: [0x54,0xda,0xff,0xf0]
            xxpermdi 7, 63, 27, 2
# CHECK-BE: xxsel 7, 63, 27, 14                # encoding: [0xf0,0xff,0xdb,0xb4]
# CHECK-LE: xxsel 7, 63, 27, 14                # encoding: [0xb4,0xdb,0xff,0xf0]
            xxsel 7, 63, 27, 14
# CHECK-BE: xxsldwi 7, 63, 27, 1               # encoding: [0xf0,0xff,0xd9,0x14]
# CHECK-LE: xxsldwi 7, 63, 27, 1               # encoding: [0x14,0xd9,0xff,0xf0]
            xxsldwi 7, 63, 27, 1
# CHECK-BE: xxspltd 7, 63, 1                   # encoding: [0xf0,0xff,0xfb,0x56]
# CHECK-LE: xxspltd 7, 63, 1                   # encoding: [0x56,0xfb,0xff,0xf0]
            xxspltd 7, 63, 1
# CHECK-BE: xxspltw 7, 27, 3                   # encoding: [0xf0,0xe3,0xda,0x90]
# CHECK-LE: xxspltw 7, 27, 3                   # encoding: [0x90,0xda,0xe3,0xf0]
            xxspltw 7, 27, 3
# CHECK-BE: xxswapd 7, 63                      # encoding: [0xf0,0xff,0xfa,0x56]
# CHECK-LE: xxswapd 7, 63                      # encoding: [0x56,0xfa,0xff,0xf0]
            xxswapd 7, 63

# Move to/from VSR
# CHECK-BE: mfvsrd 3, 40                       # encoding: [0x7d,0x03,0x00,0x67]
# CHECK-LE: mfvsrd 3, 40                       # encoding: [0x67,0x00,0x03,0x7d]
            mfvsrd 3, 40
# CHECK-BE: mfvsrd 3, 40                       # encoding: [0x7d,0x03,0x00,0x67]
# CHECK-LE: mfvsrd 3, 40                       # encoding: [0x67,0x00,0x03,0x7d]
            mfvrd 3, 8
# CHECK-BE: mfvsrwz 5, 0                       # encoding: [0x7c,0x05,0x00,0xe6]
# CHECK-LE: mfvsrwz 5, 0                       # encoding: [0xe6,0x00,0x05,0x7c]
            mfvsrwz 5, 0
# CHECK-BE: mtvsrd 0, 3                        # encoding: [0x7c,0x03,0x01,0x66]
# CHECK-LE: mtvsrd 0, 3                        # encoding: [0x66,0x01,0x03,0x7c]
            mtvsrd 0, 3
# CHECK-BE: mtvsrwa 0, 3                       # encoding: [0x7c,0x03,0x01,0xa6]
# CHECK-LE: mtvsrwa 0, 3                       # encoding: [0xa6,0x01,0x03,0x7c]
            mtvsrwa 0, 3
# CHECK-BE: mtvsrwz 0, 3                       # encoding: [0x7c,0x03,0x01,0xe6]
# CHECK-LE: mtvsrwz 0, 3                       # encoding: [0xe6,0x01,0x03,0x7c]
            mtvsrwz 0, 3

# Power9 Instructions:

# Copy Sign
# CHECK-BE: xscpsgnqp 7, 31, 27                # encoding: [0xfc,0xff,0xd8,0xc8]
# CHECK-LE: xscpsgnqp 7, 31, 27                # encoding: [0xc8,0xd8,0xff,0xfc]
            xscpsgnqp 7, 31, 27

# Absolute/Negative Absolute/Negate
# CHECK-BE: xsabsqp 7, 27                      # encoding: [0xfc,0xe0,0xde,0x48]
# CHECK-LE: xsabsqp 7, 27                      # encoding: [0x48,0xde,0xe0,0xfc]
            xsabsqp 7, 27
# CHECK-BE: xsnegqp 7, 27                      # encoding: [0xfc,0xf0,0xde,0x48]
# CHECK-LE: xsnegqp 7, 27                      # encoding: [0x48,0xde,0xf0,0xfc]
            xsnegqp 7, 27
# CHECK-BE: xsnabsqp 7, 27                     # encoding: [0xfc,0xe8,0xde,0x48]
# CHECK-LE: xsnabsqp 7, 27                     # encoding: [0x48,0xde,0xe8,0xfc]
            xsnabsqp 7, 27

# Add/Divide/Multiply/Square-Root/Subtract
# CHECK-BE: xsaddqp 7, 31, 27                  # encoding: [0xfc,0xff,0xd8,0x08]
# CHECK-LE: xsaddqp 7, 31, 27                  # encoding: [0x08,0xd8,0xff,0xfc]
            xsaddqp 7, 31, 27
# CHECK-BE: xsaddqpo 7, 31, 27                 # encoding: [0xfc,0xff,0xd8,0x09]
# CHECK-LE: xsaddqpo 7, 31, 27                 # encoding: [0x09,0xd8,0xff,0xfc]
            xsaddqpo 7, 31, 27
# CHECK-BE: xsdivqp 7, 31, 27                  # encoding: [0xfc,0xff,0xdc,0x48]
# CHECK-LE: xsdivqp 7, 31, 27                  # encoding: [0x48,0xdc,0xff,0xfc]
            xsdivqp 7, 31, 27
# CHECK-BE: xsdivqpo 7, 31, 27                 # encoding: [0xfc,0xff,0xdc,0x49]
# CHECK-LE: xsdivqpo 7, 31, 27                 # encoding: [0x49,0xdc,0xff,0xfc]
            xsdivqpo 7, 31, 27
# CHECK-BE: xsmulqp 7, 31, 27                  # encoding: [0xfc,0xff,0xd8,0x48]
# CHECK-LE: xsmulqp 7, 31, 27                  # encoding: [0x48,0xd8,0xff,0xfc]
            xsmulqp 7, 31, 27
# CHECK-BE: xsmulqpo 7, 31, 27                 # encoding: [0xfc,0xff,0xd8,0x49]
# CHECK-LE: xsmulqpo 7, 31, 27                 # encoding: [0x49,0xd8,0xff,0xfc]
            xsmulqpo 7, 31, 27
# CHECK-BE: xssqrtqp 7, 31                     # encoding: [0xfc,0xfb,0xfe,0x48]
# CHECK-LE: xssqrtqp 7, 31                     # encoding: [0x48,0xfe,0xfb,0xfc]
            xssqrtqp 7, 31
# CHECK-BE: xssqrtqpo 7, 31                    # encoding: [0xfc,0xfb,0xfe,0x49]
# CHECK-LE: xssqrtqpo 7, 31                    # encoding: [0x49,0xfe,0xfb,0xfc]
            xssqrtqpo 7, 31
# CHECK-BE: xssubqp 7, 31, 27                  # encoding: [0xfc,0xff,0xdc,0x08]
# CHECK-LE: xssubqp 7, 31, 27                  # encoding: [0x08,0xdc,0xff,0xfc]
            xssubqp 7, 31, 27
# CHECK-BE: xssubqpo 7, 31, 27                 # encoding: [0xfc,0xff,0xdc,0x09]
# CHECK-LE: xssubqpo 7, 31, 27                 # encoding: [0x09,0xdc,0xff,0xfc]
            xssubqpo 7, 31, 27

# (Negative) Multiply-Add/Subtract
# CHECK-BE: xsmaddqp 7, 31, 27                 # encoding: [0xfc,0xff,0xdb,0x08]
# CHECK-LE: xsmaddqp 7, 31, 27                 # encoding: [0x08,0xdb,0xff,0xfc]
            xsmaddqp 7, 31, 27
# CHECK-BE: xsmaddqpo 7, 31, 27                # encoding: [0xfc,0xff,0xdb,0x09]
# CHECK-LE: xsmaddqpo 7, 31, 27                # encoding: [0x09,0xdb,0xff,0xfc]
            xsmaddqpo 7, 31, 27
# CHECK-BE: xsmsubqp 7, 31, 27                 # encoding: [0xfc,0xff,0xdb,0x48]
# CHECK-LE: xsmsubqp 7, 31, 27                 # encoding: [0x48,0xdb,0xff,0xfc]
            xsmsubqp 7, 31, 27
# CHECK-BE: xsmsubqpo 7, 31, 27                # encoding: [0xfc,0xff,0xdb,0x49]
# CHECK-LE: xsmsubqpo 7, 31, 27                # encoding: [0x49,0xdb,0xff,0xfc]
            xsmsubqpo 7, 31, 27
# CHECK-BE: xsnmaddqp 7, 31, 27                # encoding: [0xfc,0xff,0xdb,0x88]
# CHECK-LE: xsnmaddqp 7, 31, 27                # encoding: [0x88,0xdb,0xff,0xfc]
            xsnmaddqp 7, 31, 27
# CHECK-BE: xsnmaddqpo 7, 31, 27               # encoding: [0xfc,0xff,0xdb,0x89]
# CHECK-LE: xsnmaddqpo 7, 31, 27               # encoding: [0x89,0xdb,0xff,0xfc]
            xsnmaddqpo 7, 31, 27
# CHECK-BE: xsnmsubqp 7, 31, 27                # encoding: [0xfc,0xff,0xdb,0xc8]
# CHECK-LE: xsnmsubqp 7, 31, 27                # encoding: [0xc8,0xdb,0xff,0xfc]
            xsnmsubqp 7, 31, 27
# CHECK-BE: xsnmsubqpo 7, 31, 27               # encoding: [0xfc,0xff,0xdb,0xc9]
# CHECK-LE: xsnmsubqpo 7, 31, 27               # encoding: [0xc9,0xdb,0xff,0xfc]
            xsnmsubqpo 7, 31, 27

# Compare Ordered/Unordered
# CHECK-BE: xscmpoqp 6, 31, 27                 # encoding: [0xff,0x1f,0xd9,0x08]
# CHECK-LE: xscmpoqp 6, 31, 27                 # encoding: [0x08,0xd9,0x1f,0xff]
            xscmpoqp 6, 31, 27
# CHECK-BE: xscmpuqp 6, 31, 27                 # encoding: [0xff,0x1f,0xdd,0x08]
# CHECK-LE: xscmpuqp 6, 31, 27                 # encoding: [0x08,0xdd,0x1f,0xff]
            xscmpuqp 6, 31, 27

# Compare Exponents
# CHECK-BE: xscmpexpdp 6, 63, 27               # encoding: [0xf3,0x1f,0xd9,0xdc]
# CHECK-LE: xscmpexpdp 6, 63, 27               # encoding: [0xdc,0xd9,0x1f,0xf3]
            xscmpexpdp 6, 63, 27
# CHECK-BE: xscmpexpqp 6, 31, 27               # encoding: [0xff,0x1f,0xd9,0x48]
# CHECK-LE: xscmpexpqp 6, 31, 27               # encoding: [0x48,0xd9,0x1f,0xff]
            xscmpexpqp 6, 31, 27

# Compare ==, >=, >
# CHECK-BE: xscmpeqdp 7, 63, 27                # encoding: [0xf0,0xff,0xd8,0x1c]
# CHECK-LE: xscmpeqdp 7, 63, 27                # encoding: [0x1c,0xd8,0xff,0xf0]
            xscmpeqdp 7, 63, 27
# CHECK-BE: xscmpgedp 7, 63, 27                # encoding: [0xf0,0xff,0xd8,0x9c]
# CHECK-LE: xscmpgedp 7, 63, 27                # encoding: [0x9c,0xd8,0xff,0xf0]
            xscmpgedp 7, 63, 27
# CHECK-BE: xscmpgtdp 7, 63, 27                # encoding: [0xf0,0xff,0xd8,0x5c]
# CHECK-LE: xscmpgtdp 7, 63, 27                # encoding: [0x5c,0xd8,0xff,0xf0]
            xscmpgtdp 7, 63, 27

# Convert DP -> QP
# CHECK-BE: xscvdpqp 7, 27                     # encoding: [0xfc,0xf6,0xde,0x88]
# CHECK-LE: xscvdpqp 7, 27                     # encoding: [0x88,0xde,0xf6,0xfc]
            xscvdpqp 7, 27

# Round & Convert QP -> DP
# CHECK-BE: xscvqpdp 7, 27                     # encoding: [0xfc,0xf4,0xde,0x88]
# CHECK-LE: xscvqpdp 7, 27                     # encoding: [0x88,0xde,0xf4,0xfc]
            xscvqpdp 7, 27
# CHECK-BE: xscvqpdpo 7, 27                    # encoding: [0xfc,0xf4,0xde,0x89]
# CHECK-LE: xscvqpdpo 7, 27                    # encoding: [0x89,0xde,0xf4,0xfc]
            xscvqpdpo 7, 27

# Truncate & Convert QP -> (Un)Signed (D)Word
# CHECK-BE: xscvqpsdz 7, 27                    # encoding: [0xfc,0xf9,0xde,0x88]
# CHECK-LE: xscvqpsdz 7, 27                    # encoding: [0x88,0xde,0xf9,0xfc]
            xscvqpsdz 7, 27
# CHECK-BE: xscvqpswz 7, 27                    # encoding: [0xfc,0xe9,0xde,0x88]
# CHECK-LE: xscvqpswz 7, 27                    # encoding: [0x88,0xde,0xe9,0xfc]
            xscvqpswz 7, 27
# CHECK-BE: xscvqpudz 7, 27                    # encoding: [0xfc,0xf1,0xde,0x88]
# CHECK-LE: xscvqpudz 7, 27                    # encoding: [0x88,0xde,0xf1,0xfc]
            xscvqpudz 7, 27
# CHECK-BE: xscvqpuwz 7, 27                    # encoding: [0xfc,0xe1,0xde,0x88]
# CHECK-LE: xscvqpuwz 7, 27                    # encoding: [0x88,0xde,0xe1,0xfc]
            xscvqpuwz 7, 27

# Convert (Un)Signed DWord -> QP
# CHECK-BE: xscvsdqp 7, 27                     # encoding: [0xfc,0xea,0xde,0x88]
# CHECK-LE: xscvsdqp 7, 27                     # encoding: [0x88,0xde,0xea,0xfc]
            xscvsdqp 7, 27
# CHECK-BE: xscvudqp 7, 27                     # encoding: [0xfc,0xe2,0xde,0x88]
# CHECK-LE: xscvudqp 7, 27                     # encoding: [0x88,0xde,0xe2,0xfc]
            xscvudqp 7, 27

# (Round &) Convert DP <-> HP
# CHECK-BE: xscvdphp 7, 63                     # encoding: [0xf0,0xf1,0xfd,0x6e]
# CHECK-LE: xscvdphp 7, 63                     # encoding: [0x6e,0xfd,0xf1,0xf0]
            xscvdphp 7, 63
# CHECK-BE: xscvhpdp 7, 63                     # encoding: [0xf0,0xf0,0xfd,0x6e]
# CHECK-LE: xscvhpdp 7, 63                     # encoding: [0x6e,0xfd,0xf0,0xf0]
            xscvhpdp 7, 63

# HP -> SP
# CHECK-BE: xvcvhpsp 7, 63                     # encoding: [0xf0,0xf8,0xff,0x6e]
# CHECK-LE: xvcvhpsp 7, 63                     # encoding: [0x6e,0xff,0xf8,0xf0]
            xvcvhpsp 7, 63
# CHECK-BE: xvcvsphp 7, 63                     # encoding: [0xf0,0xf9,0xff,0x6e]
# CHECK-LE: xvcvsphp 7, 63                     # encoding: [0x6e,0xff,0xf9,0xf0]
            xvcvsphp 7, 63

# Round to Quad-Precision Integer [with Inexact]
# CHECK-BE: xsrqpi 1, 7, 27, 2                 # encoding: [0xfc,0xe1,0xdc,0x0a]
# CHECK-LE: xsrqpi 1, 7, 27, 2                 # encoding: [0x0a,0xdc,0xe1,0xfc]
            xsrqpi 1, 7, 27, 2
# CHECK-BE: xsrqpix 1, 7, 27, 2                # encoding: [0xfc,0xe1,0xdc,0x0b]
# CHECK-LE: xsrqpix 1, 7, 27, 2                # encoding: [0x0b,0xdc,0xe1,0xfc]
            xsrqpix 1, 7, 27, 2

# Round Quad-Precision to Double-Extended Precision
# CHECK-BE: xsrqpxp 1, 7, 27, 2                # encoding: [0xfc,0xe1,0xdc,0x4a]
# CHECK-LE: xsrqpxp 1, 7, 27, 2                # encoding: [0x4a,0xdc,0xe1,0xfc]
            xsrqpxp 1, 7, 27, 2

# Insert Exponent DP/QP
# CHECK-BE: xsiexpdp 63, 3, 4                  # encoding: [0xf3,0xe3,0x27,0x2d]
# CHECK-LE: xsiexpdp 63, 3, 4                  # encoding: [0x2d,0x27,0xe3,0xf3]
            xsiexpdp 63, 3, 4
# CHECK-BE: xsiexpqp 7, 31, 27                 # encoding: [0xfc,0xff,0xde,0xc8]
# CHECK-LE: xsiexpqp 7, 31, 27                 # encoding: [0xc8,0xde,0xff,0xfc]
            xsiexpqp 7, 31, 27

# Vector Insert Exponent DP
# CHECK-BE: xviexpdp 7, 63, 27                 # encoding: [0xf0,0xff,0xdf,0xc4]
# CHECK-LE: xviexpdp 7, 63, 27                 # encoding: [0xc4,0xdf,0xff,0xf0]
            xviexpdp 7, 63, 27
# CHECK-BE: xviexpsp 7, 63, 27                 # encoding: [0xf0,0xff,0xde,0xc4]
# CHECK-LE: xviexpsp 7, 63, 27                 # encoding: [0xc4,0xde,0xff,0xf0]
            xviexpsp 7, 63, 27

# Vector Extract Unsigned Word
# CHECK-BE: xxextractuw 7, 63, 15              # encoding: [0xf0,0xef,0xfa,0x96]
# CHECK-LE: xxextractuw 7, 63, 15              # encoding: [0x96,0xfa,0xef,0xf0]
            xxextractuw 7, 63, 15

# Vector Insert Word
# CHECK-BE: xxinsertw 7, 63, 15                # encoding: [0xf0,0xef,0xfa,0xd6]
# CHECK-LE: xxinsertw 7, 63, 15                # encoding: [0xd6,0xfa,0xef,0xf0]
            xxinsertw 7, 63, 15

# Extract Exponent/Significand DP/QP
# CHECK-BE: xsxexpdp 7, 63                     # encoding: [0xf0,0xe0,0xfd,0x6e]
# CHECK-LE: xsxexpdp 7, 63                     # encoding: [0x6e,0xfd,0xe0,0xf0]
            xsxexpdp 7, 63
# CHECK-BE: xsxsigdp 7, 63                     # encoding: [0xf0,0xe1,0xfd,0x6e]
# CHECK-LE: xsxsigdp 7, 63                     # encoding: [0x6e,0xfd,0xe1,0xf0]
            xsxsigdp 7, 63
# CHECK-BE: xsxexpqp 7, 31                     # encoding: [0xfc,0xe2,0xfe,0x48]
# CHECK-LE: xsxexpqp 7, 31                     # encoding: [0x48,0xfe,0xe2,0xfc]
            xsxexpqp 7, 31
# CHECK-BE: xsxsigqp 7, 31                     # encoding: [0xfc,0xf2,0xfe,0x48]
# CHECK-LE: xsxsigqp 7, 31                     # encoding: [0x48,0xfe,0xf2,0xfc]
            xsxsigqp 7, 31

# Vector Extract Exponent/Significand DP
# CHECK-BE: xvxexpdp 7, 63                     # encoding: [0xf0,0xe0,0xff,0x6e]
# CHECK-LE: xvxexpdp 7, 63                     # encoding: [0x6e,0xff,0xe0,0xf0]
            xvxexpdp 7, 63
# CHECK-BE: xvxexpsp 7, 63                     # encoding: [0xf0,0xe8,0xff,0x6e]
# CHECK-LE: xvxexpsp 7, 63                     # encoding: [0x6e,0xff,0xe8,0xf0]
            xvxexpsp 7, 63
# CHECK-BE: xvxsigdp 7, 63                     # encoding: [0xf0,0xe1,0xff,0x6e]
# CHECK-LE: xvxsigdp 7, 63                     # encoding: [0x6e,0xff,0xe1,0xf0]
            xvxsigdp 7, 63
# CHECK-BE: xvxsigsp 7, 63                     # encoding: [0xf0,0xe9,0xff,0x6e]
# CHECK-LE: xvxsigsp 7, 63                     # encoding: [0x6e,0xff,0xe9,0xf0]
            xvxsigsp 7, 63

# Test Data Class SP/DP/QP
# CHECK-BE: xststdcsp 7, 63, 127               # encoding: [0xf3,0xff,0xfc,0xaa]
# CHECK-LE: xststdcsp 7, 63, 127               # encoding: [0xaa,0xfc,0xff,0xf3]
            xststdcsp 7, 63, 127
# CHECK-BE: xststdcdp 7, 63, 127               # encoding: [0xf3,0xff,0xfd,0xaa]
# CHECK-LE: xststdcdp 7, 63, 127               # encoding: [0xaa,0xfd,0xff,0xf3]
            xststdcdp 7, 63, 127
# CHECK-BE: xststdcqp 7, 31, 127               # encoding: [0xff,0xff,0xfd,0x88]
# CHECK-LE: xststdcqp 7, 31, 127               # encoding: [0x88,0xfd,0xff,0xff]
            xststdcqp 7, 31, 127

# Vector Test Data Class SP/DP
# CHECK-BE: xststdcsp 7, 63, 127               # encoding: [0xf3,0xff,0xfc,0xaa]
# CHECK-LE: xststdcsp 7, 63, 127               # encoding: [0xaa,0xfc,0xff,0xf3]
            xststdcsp 7, 63, 127
# CHECK-BE: xststdcdp 7, 63, 127               # encoding: [0xf3,0xff,0xfd,0xaa]
# CHECK-LE: xststdcdp 7, 63, 127               # encoding: [0xaa,0xfd,0xff,0xf3]
            xststdcdp 7, 63, 127
# CHECK-BE: xststdcqp 7, 31, 127               # encoding: [0xff,0xff,0xfd,0x88]
# CHECK-LE: xststdcqp 7, 31, 127               # encoding: [0x88,0xfd,0xff,0xff]
            xststdcqp 7, 31, 127

# Maximum/Minimum Type-C/Type-J DP
# CHECK-BE: xsmaxcdp 7, 63, 27                 # encoding: [0xf0,0xff,0xdc,0x04]
# CHECK-LE: xsmaxcdp 7, 63, 27                 # encoding: [0x04,0xdc,0xff,0xf0]
            xsmaxcdp 7, 63, 27
# CHECK-BE: xsmaxjdp 7, 63, 27                 # encoding: [0xf0,0xff,0xdc,0x84]
# CHECK-LE: xsmaxjdp 7, 63, 27                 # encoding: [0x84,0xdc,0xff,0xf0]
            xsmaxjdp 7, 63, 27
# CHECK-BE: xsmincdp 7, 63, 27                 # encoding: [0xf0,0xff,0xdc,0x44]
# CHECK-LE: xsmincdp 7, 63, 27                 # encoding: [0x44,0xdc,0xff,0xf0]
            xsmincdp 7, 63, 27
# CHECK-BE: xsminjdp 7, 63, 27                 # encoding: [0xf0,0xff,0xdc,0xc4]
# CHECK-LE: xsminjdp 7, 63, 27                 # encoding: [0xc4,0xdc,0xff,0xf0]
            xsminjdp 7, 63, 27

# Vector Byte-Reverse H/W/D/Q Word
# CHECK-BE: xxbrh 7, 63                        # encoding: [0xf0,0xe7,0xff,0x6e]
# CHECK-LE: xxbrh 7, 63                        # encoding: [0x6e,0xff,0xe7,0xf0]
            xxbrh 7, 63
# CHECK-BE: xxbrw 7, 63                        # encoding: [0xf0,0xef,0xff,0x6e]
# CHECK-LE: xxbrw 7, 63                        # encoding: [0x6e,0xff,0xef,0xf0]
            xxbrw 7, 63
# CHECK-BE: xxbrd 7, 63                        # encoding: [0xf0,0xf7,0xff,0x6e]
# CHECK-LE: xxbrd 7, 63                        # encoding: [0x6e,0xff,0xf7,0xf0]
            xxbrd 7, 63
# CHECK-BE: xxbrq 7, 63                        # encoding: [0xf0,0xff,0xff,0x6e]
# CHECK-LE: xxbrq 7, 63                        # encoding: [0x6e,0xff,0xff,0xf0]
            xxbrq 7, 63

# Vector Permute
# CHECK-BE: xxperm 7, 63, 27                   # encoding: [0xf0,0xff,0xd8,0xd4]
# CHECK-LE: xxperm 7, 63, 27                   # encoding: [0xd4,0xd8,0xff,0xf0]
            xxperm 7, 63, 27
# CHECK-BE: xxpermr 7, 63, 27                  # encoding: [0xf0,0xff,0xd9,0xd4]
# CHECK-LE: xxpermr 7, 63, 27                  # encoding: [0xd4,0xd9,0xff,0xf0]
            xxpermr 7, 63, 27

# Vector Splat Immediate Byte
# CHECK-BE: xxspltib 63, 255                   # encoding: [0xf3,0xe7,0xfa,0xd1]
# CHECK-LE: xxspltib 63, 255                   # encoding: [0xd1,0xfa,0xe7,0xf3]
            xxspltib 63, 255

# Load/Store Vector, test maximum and minimum displacement value
# CHECK-BE: lxv 61, 32752(31)                  # encoding: [0xf7,0xbf,0x7f,0xf9]
# CHECK-LE: lxv 61, 32752(31)                  # encoding: [0xf9,0x7f,0xbf,0xf7]
            lxv 61, 32752(31)
# CHECK-BE: lxv 61, -32768(0)                  # encoding: [0xf7,0xa0,0x80,0x09]
# CHECK-LE: lxv 61, -32768(0)                  # encoding: [0x09,0x80,0xa0,0xf7]
            lxv 61, -32768(0)
# CHECK-BE: stxv 61, 32752(31)                 # encoding: [0xf7,0xbf,0x7f,0xfd]
# CHECK-LE: stxv 61, 32752(31)                 # encoding: [0xfd,0x7f,0xbf,0xf7]
            stxv 61, 32752(31)
# CHECK-BE: stxv 61, -32768(0)                 # encoding: [0xf7,0xa0,0x80,0x0d]
# CHECK-LE: stxv 61, -32768(0)                 # encoding: [0x0d,0x80,0xa0,0xf7]
            stxv 61, -32768(0)

# Load/Store DWord
# CHECK-BE: lxsd 31, -32768(0)                 # encoding: [0xe7,0xe0,0x80,0x02]
# CHECK-LE: lxsd 31, -32768(0)                 # encoding: [0x02,0x80,0xe0,0xe7]
            lxsd 31, -32768(0)
# CHECK-BE: lxsd 31, 32764(12)                 # encoding: [0xe7,0xec,0x7f,0xfe]
# CHECK-LE: lxsd 31, 32764(12)                 # encoding: [0xfe,0x7f,0xec,0xe7]
            lxsd 31, 32764(12)
# CHECK-BE: stxsd 31, 32764(12)                # encoding: [0xf7,0xec,0x7f,0xfe]
# CHECK-LE: stxsd 31, 32764(12)                # encoding: [0xfe,0x7f,0xec,0xf7]
            stxsd 31, 32764(12)

# Load SP from src, convert it to DP, and place in dword[0]
# CHECK-BE: lxssp 31, -32768(0)                # encoding: [0xe7,0xe0,0x80,0x03]
# CHECK-LE: lxssp 31, -32768(0)                # encoding: [0x03,0x80,0xe0,0xe7]
            lxssp 31, -32768(0)
# CHECK-BE: lxssp 31, 32764(12)                # encoding: [0xe7,0xec,0x7f,0xff]
# CHECK-LE: lxssp 31, 32764(12)                # encoding: [0xff,0x7f,0xec,0xe7]
            lxssp 31, 32764(12)

# Convert DP of dword[0] to SP, and Store to dst
# CHECK-BE: stxssp 31, -32768(0)               # encoding: [0xf7,0xe0,0x80,0x03]
# CHECK-LE: stxssp 31, -32768(0)               # encoding: [0x03,0x80,0xe0,0xf7]
            stxssp 31, -32768(0)

# Load as Integer Byte/Halfword & Zero Indexed
# CHECK-BE: lxsibzx 57, 12, 27                 # encoding: [0x7f,0x2c,0xde,0x1b]
# CHECK-LE: lxsibzx 57, 12, 27                 # encoding: [0x1b,0xde,0x2c,0x7f]
            lxsibzx 57, 12, 27
# CHECK-BE: lxsihzx 57, 12, 27                 # encoding: [0x7f,0x2c,0xde,0x5b]
# CHECK-LE: lxsihzx 57, 12, 27                 # encoding: [0x5b,0xde,0x2c,0x7f]
            lxsihzx 57, 12, 27

# Load Vector Halfword*8/Byte*16 Indexed
# CHECK-BE: lxvh8x 57, 12, 27                  # encoding: [0x7f,0x2c,0xde,0x59]
# CHECK-LE: lxvh8x 57, 12, 27                  # encoding: [0x59,0xde,0x2c,0x7f]
            lxvh8x 57, 12, 27
# CHECK-BE: lxvb16x 57, 12, 27                 # encoding: [0x7f,0x2c,0xde,0xd9]
# CHECK-LE: lxvb16x 57, 12, 27                 # encoding: [0xd9,0xde,0x2c,0x7f]
            lxvb16x 57, 12, 27

# Load Vector Indexed
# CHECK-BE: lxvx 57, 12, 27                    # encoding: [0x7f,0x2c,0xda,0x19]
# CHECK-LE: lxvx 57, 12, 27                    # encoding: [0x19,0xda,0x2c,0x7f]
            lxvx 57, 12, 27

# Load Vector (Left-justified) with Length
# CHECK-BE: lxvl 57, 12, 27                    # encoding: [0x7f,0x2c,0xda,0x1b]
# CHECK-LE: lxvl 57, 12, 27                    # encoding: [0x1b,0xda,0x2c,0x7f]
            lxvl 57, 12, 27
# CHECK-BE: lxvll 57, 12, 27                   # encoding: [0x7f,0x2c,0xda,0x5b]
# CHECK-LE: lxvll 57, 12, 27                   # encoding: [0x5b,0xda,0x2c,0x7f]
            lxvll 57, 12, 27

# Load Vector Word & Splat Indexed
# CHECK-BE: lxvwsx 57, 12, 27                  # encoding: [0x7f,0x2c,0xda,0xd9]
# CHECK-LE: lxvwsx 57, 12, 27                  # encoding: [0xd9,0xda,0x2c,0x7f]
            lxvwsx 57, 12, 27

# Store as Integer Byte/Halfword Indexed
# CHECK-BE: stxsibx 57, 12, 27                 # encoding: [0x7f,0x2c,0xdf,0x1b]
# CHECK-LE: stxsibx 57, 12, 27                 # encoding: [0x1b,0xdf,0x2c,0x7f]
            stxsibx 57, 12, 27
# CHECK-BE: stxsihx 57, 12, 27                 # encoding: [0x7f,0x2c,0xdf,0x5b]
# CHECK-LE: stxsihx 57, 12, 27                 # encoding: [0x5b,0xdf,0x2c,0x7f]
            stxsihx 57, 12, 27

# Store Vector Halfword*8/Byte*16 Indexed
# CHECK-BE: stxvh8x 57, 12, 27                 # encoding: [0x7f,0x2c,0xdf,0x59]
# CHECK-LE: stxvh8x 57, 12, 27                 # encoding: [0x59,0xdf,0x2c,0x7f]
            stxvh8x 57, 12, 27
# CHECK-BE: stxvb16x 57, 12, 27                # encoding: [0x7f,0x2c,0xdf,0xd9]
# CHECK-LE: stxvb16x 57, 12, 27                # encoding: [0xd9,0xdf,0x2c,0x7f]
            stxvb16x 57, 12, 27

# Store Vector Indexed
# CHECK-BE: stxvx 57, 12, 27                   # encoding: [0x7f,0x2c,0xdb,0x19]
# CHECK-LE: stxvx 57, 12, 27                   # encoding: [0x19,0xdb,0x2c,0x7f]
            stxvx 57, 12, 27

# Store Vector (Left-justified) with Length
# CHECK-BE: stxvl 57, 12, 27                   # encoding: [0x7f,0x2c,0xdb,0x1b]
# CHECK-LE: stxvl 57, 12, 27                   # encoding: [0x1b,0xdb,0x2c,0x7f]
            stxvl 57, 12, 27
# CHECK-BE: stxvll 57, 12, 27                  # encoding: [0x7f,0x2c,0xdb,0x5b]
# CHECK-LE: stxvll 57, 12, 27                  # encoding: [0x5b,0xdb,0x2c,0x7f]
            stxvll 57, 12, 27

# P9 Direct Move Instructions
# CHECK-BE: mtvsrws 34, 3                      # encoding: [0x7c,0x43,0x03,0x27]
# CHECK-LE: mtvsrws 34, 3                      # encoding: [0x27,0x03,0x43,0x7c]
            mtvsrws 34, 3

# CHECK-BE: mtvsrdd 34, 3, 12                  # encoding: [0x7c,0x43,0x63,0x67]
# CHECK-LE: mtvsrdd 34, 3, 12                  # encoding: [0x67,0x63,0x43,0x7c]
            mtvsrdd 34, 3, 12

# CHECK-BE: mfvsrld 3, 34                      # encoding: [0x7c,0x43,0x02,0x67]
# CHECK-LE: mfvsrld 3, 34                      # encoding: [0x67,0x02,0x43,0x7c]
            mfvsrld 3, 34

# CHECK-BE: xvtstdcdp 63, 63, 65               # encoding: [0xf3,0xe1,0xff,0xeb]
# CHECK-LE: xvtstdcdp 63, 63, 65               # encoding: [0xeb,0xff,0xe1,0xf3]
            xvtstdcdp 63, 63, 65
# CHECK-BE: xvtstdcsp 63, 63, 34               # encoding: [0xf3,0xe2,0xfe,0xaf]
# CHECK-LE: xvtstdcsp 63, 63, 34               # encoding: [0xaf,0xfe,0xe2,0xf3]
            xvtstdcsp 63, 63, 34
