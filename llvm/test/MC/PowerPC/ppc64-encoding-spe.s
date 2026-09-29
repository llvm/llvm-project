# RUN: llvm-mc -triple powerpc64-unknown-unknown --show-encoding %s | FileCheck -check-prefix=CHECK-BE %s
# RUN: llvm-mc -triple powerpc64le-unknown-unknown --show-encoding %s | FileCheck -check-prefix=CHECK-LE %s

# Instructions from the Signal Processing Engine extension:

# CHECK-BE: evlddx 14, 21, 28               # encoding: [0x11,0xd5,0xe3,0x00]
# CHECK-LE: evlddx 14, 21, 28               # encoding: [0x00,0xe3,0xd5,0x11]
            evlddx %r14, %r21, %r28
# CHECK-BE: evldwx 14, 21, 28               # encoding: [0x11,0xd5,0xe3,0x02]
# CHECK-LE: evldwx 14, 21, 28               # encoding: [0x02,0xe3,0xd5,0x11]
            evldwx %r14, %r21, %r28
# CHECK-BE: evldhx 14, 21, 28               # encoding: [0x11,0xd5,0xe3,0x04]
# CHECK-LE: evldhx 14, 21, 28               # encoding: [0x04,0xe3,0xd5,0x11]
            evldhx %r14, %r21, %r28
# CHECK-BE: evlhhesplatx 14, 21, 28         # encoding: [0x11,0xd5,0xe3,0x08]
# CHECK-LE: evlhhesplatx 14, 21, 28         # encoding: [0x08,0xe3,0xd5,0x11]
            evlhhesplatx %r14, %r21, %r28
# CHECK-BE: evlhhousplatx 14, 21, 28        # encoding: [0x11,0xd5,0xe3,0x0c]
# CHECK-LE: evlhhousplatx 14, 21, 28        # encoding: [0x0c,0xe3,0xd5,0x11]
            evlhhousplatx %r14, %r21, %r28
# CHECK-BE: evlhhossplatx 14, 21, 28        # encoding: [0x11,0xd5,0xe3,0x0e]
# CHECK-LE: evlhhossplatx 14, 21, 28        # encoding: [0x0e,0xe3,0xd5,0x11]
            evlhhossplatx %r14, %r21, %r28
# CHECK-BE: evlwhex 14, 21, 28              # encoding: [0x11,0xd5,0xe3,0x10]
# CHECK-LE: evlwhex 14, 21, 28              # encoding: [0x10,0xe3,0xd5,0x11]
            evlwhex %r14, %r21, %r28
# CHECK-BE: evlwhoux 14, 21, 28             # encoding: [0x11,0xd5,0xe3,0x14]
# CHECK-LE: evlwhoux 14, 21, 28             # encoding: [0x14,0xe3,0xd5,0x11]
            evlwhoux %r14, %r21, %r28
# CHECK-BE: evlwhosx 14, 21, 28             # encoding: [0x11,0xd5,0xe3,0x16]
# CHECK-LE: evlwhosx 14, 21, 28             # encoding: [0x16,0xe3,0xd5,0x11]
            evlwhosx %r14, %r21, %r28
# CHECK-BE: evlwwsplatx 14, 21, 28          # encoding: [0x11,0xd5,0xe3,0x18]
# CHECK-LE: evlwwsplatx 14, 21, 28          # encoding: [0x18,0xe3,0xd5,0x11]
            evlwwsplatx %r14, %r21, %r28
# CHECK-BE: evlwhsplatx 14, 21, 28          # encoding: [0x11,0xd5,0xe3,0x1c]
# CHECK-LE: evlwhsplatx 14, 21, 28          # encoding: [0x1c,0xe3,0xd5,0x11]
            evlwhsplatx %r14, %r21, %r28
# CHECK-BE: evmergehi 14, 21, 28            # encoding: [0x11,0xd5,0xe2,0x2c]
# CHECK-LE: evmergehi 14, 21, 28            # encoding: [0x2c,0xe2,0xd5,0x11]
            evmergehi %r14, %r21, %r28
# CHECK-BE: evmergelo 14, 21, 28            # encoding: [0x11,0xd5,0xe2,0x2d]
# CHECK-LE: evmergelo 14, 21, 28            # encoding: [0x2d,0xe2,0xd5,0x11]
            evmergelo %r14, %r21, %r28
# CHECK-BE: evmergehilo 14, 21, 28          # encoding: [0x11,0xd5,0xe2,0x2e]
# CHECK-LE: evmergehilo 14, 21, 28          # encoding: [0x2e,0xe2,0xd5,0x11]
            evmergehilo %r14, %r21, %r28
# CHECK-BE: evmergelohi 14, 21, 28          # encoding: [0x11,0xd5,0xe2,0x2f]
# CHECK-LE: evmergelohi 14, 21, 28          # encoding: [0x2f,0xe2,0xd5,0x11]
            evmergelohi %r14, %r21, %r28

# CHECK-BE: brinc 14, 22, 19                # encoding: [0x11,0xd6,0x9a,0x0f]
# CHECK-LE: brinc 14, 22, 19                # encoding: [0x0f,0x9a,0xd6,0x11]
            brinc %r14, %r22, %r19
# CHECK-BE: evabs 14, 22                    # encoding: [0x11,0xd6,0x02,0x08]
# CHECK-LE: evabs 14, 22                    # encoding: [0x08,0x02,0xd6,0x11]
            evabs %r14, %r22
# CHECK-BE: evaddsmiaaw 14, 22              # encoding: [0x11,0xd6,0x04,0xc9]
# CHECK-LE: evaddsmiaaw 14, 22              # encoding: [0xc9,0x04,0xd6,0x11]
            evaddsmiaaw %r14, %r22
# CHECK-BE: evaddssiaaw 14, 22              # encoding: [0x11,0xd6,0x04,0xc1]
# CHECK-LE: evaddssiaaw 14, 22              # encoding: [0xc1,0x04,0xd6,0x11]
            evaddssiaaw %r14, %r22
# CHECK-BE: evaddusiaaw 14, 22              # encoding: [0x11,0xd6,0x04,0xc0]
# CHECK-LE: evaddusiaaw 14, 22              # encoding: [0xc0,0x04,0xd6,0x11]
            evaddusiaaw %r14, %r22
# CHECK-BE: evaddumiaaw 14, 22              # encoding: [0x11,0xd6,0x04,0xc8]
# CHECK-LE: evaddumiaaw 14, 22              # encoding: [0xc8,0x04,0xd6,0x11]
            evaddumiaaw %r14, %r22
# CHECK-BE: evaddw 14, 22, 19               # encoding: [0x11,0xd6,0x9a,0x00]
# CHECK-LE: evaddw 14, 22, 19               # encoding: [0x00,0x9a,0xd6,0x11]
            evaddw %r14, %r22, %r19
# CHECK-BE: evaddiw 14, 22, 19              # encoding: [0x11,0xd3,0xb2,0x02]
# CHECK-LE: evaddiw 14, 22, 19              # encoding: [0x02,0xb2,0xd3,0x11]
            evaddiw %r14, %r22, 19
# CHECK-BE: evand 14, 22, 19                # encoding: [0x11,0xd6,0x9a,0x11]
# CHECK-LE: evand 14, 22, 19                # encoding: [0x11,0x9a,0xd6,0x11]
            evand %r14, %r22, %r19
# CHECK-BE: evandc 14, 22, 19               # encoding: [0x11,0xd6,0x9a,0x12]
# CHECK-LE: evandc 14, 22, 19               # encoding: [0x12,0x9a,0xd6,0x11]
            evandc %r14, %r22, %r19
# CHECK-BE: evcmpeq 3, 22, 19            # encoding: [0x11,0x96,0x9a,0x34]
# CHECK-LE: evcmpeq 3, 22, 19            # encoding: [0x34,0x9a,0x96,0x11]
            evcmpeq %cr3, %r22, %r19
# CHECK-BE: evcmpgts 3, 22, 19           # encoding: [0x11,0x96,0x9a,0x31]
# CHECK-LE: evcmpgts 3, 22, 19           # encoding: [0x31,0x9a,0x96,0x11]
            evcmpgts %cr3, %r22, %r19
# CHECK-BE: evcmpgtu 3, 22, 19           # encoding: [0x11,0x96,0x9a,0x30]
# CHECK-LE: evcmpgtu 3, 22, 19           # encoding: [0x30,0x9a,0x96,0x11]
            evcmpgtu %cr3, %r22, %r19
# CHECK-BE: evcmplts 3, 22, 19           # encoding: [0x11,0x96,0x9a,0x33]
# CHECK-LE: evcmplts 3, 22, 19           # encoding: [0x33,0x9a,0x96,0x11]
            evcmplts %cr3, %r22, %r19
# CHECK-BE: evcmpltu 3, 22, 19           # encoding: [0x11,0x96,0x9a,0x32]
# CHECK-LE: evcmpltu 3, 22, 19           # encoding: [0x32,0x9a,0x96,0x11]
            evcmpltu %cr3, %r22, %r19
# CHECK-BE: evcntlsw 14, 22                 # encoding: [0x11,0xd6,0x02,0x0e]
# CHECK-LE: evcntlsw 14, 22                 # encoding: [0x0e,0x02,0xd6,0x11]
            evcntlsw %r14, %r22
# CHECK-BE: evcntlzw 14, 22                 # encoding: [0x11,0xd6,0x02,0x0d]
# CHECK-LE: evcntlzw 14, 22                 # encoding: [0x0d,0x02,0xd6,0x11]
            evcntlzw %r14, %r22
# CHECK-BE: evdivws 14, 22, 19              # encoding: [0x11,0xd6,0x9c,0xc6]
# CHECK-LE: evdivws 14, 22, 19              # encoding: [0xc6,0x9c,0xd6,0x11]
            evdivws %r14, %r22, %r19
# CHECK-BE: evdivwu 14, 22, 19              # encoding: [0x11,0xd6,0x9c,0xc7]
# CHECK-LE: evdivwu 14, 22, 19              # encoding: [0xc7,0x9c,0xd6,0x11]
            evdivwu %r14, %r22, %r19
# CHECK-BE: eveqv 14, 22, 19                # encoding: [0x11,0xd6,0x9a,0x19]
# CHECK-LE: eveqv 14, 22, 19                # encoding: [0x19,0x9a,0xd6,0x11]
            eveqv %r14, %r22, %r19
# CHECK-BE: evextsb 14, 22                  # encoding: [0x11,0xd6,0x02,0x0a]
# CHECK-LE: evextsb 14, 22                  # encoding: [0x0a,0x02,0xd6,0x11]
            evextsb %r14, %r22
# CHECK-BE: evextsh 14, 22                  # encoding: [0x11,0xd6,0x02,0x0b]
# CHECK-LE: evextsh 14, 22                  # encoding: [0x0b,0x02,0xd6,0x11]
            evextsh %r14, %r22
# CHECK-BE: evmhegsmfaa 14, 22, 19          # encoding: [0x11,0xd6,0x9d,0x2b]
# CHECK-LE: evmhegsmfaa 14, 22, 19          # encoding: [0x2b,0x9d,0xd6,0x11]
            evmhegsmfaa %r14, %r22, %r19
# CHECK-BE: evmhegsmfan 14, 22, 19          # encoding: [0x11,0xd6,0x9d,0xab]
# CHECK-LE: evmhegsmfan 14, 22, 19          # encoding: [0xab,0x9d,0xd6,0x11]
            evmhegsmfan %r14, %r22, %r19
# CHECK-BE: evmhegsmiaa 14, 22, 19          # encoding: [0x11,0xd6,0x9d,0x29]
# CHECK-LE: evmhegsmiaa 14, 22, 19          # encoding: [0x29,0x9d,0xd6,0x11]
            evmhegsmiaa %r14, %r22, %r19
# CHECK-BE: evmhegsmian 14, 22, 19          # encoding: [0x11,0xd6,0x9d,0xa9]
# CHECK-LE: evmhegsmian 14, 22, 19          # encoding: [0xa9,0x9d,0xd6,0x11]
            evmhegsmian %r14, %r22, %r19
# CHECK-BE: evmhegumiaa 14, 22, 19          # encoding: [0x11,0xd6,0x9d,0x28]
# CHECK-LE: evmhegumiaa 14, 22, 19          # encoding: [0x28,0x9d,0xd6,0x11]
            evmhegumiaa %r14, %r22, %r19
# CHECK-BE: evmhegumian 14, 22, 19          # encoding: [0x11,0xd6,0x9d,0xa8]
# CHECK-LE: evmhegumian 14, 22, 19          # encoding: [0xa8,0x9d,0xd6,0x11]
            evmhegumian %r14, %r22, %r19
# CHECK-BE: evmhesmf 14, 22, 19             # encoding: [0x11,0xd6,0x9c,0x0b]
# CHECK-LE: evmhesmf 14, 22, 19             # encoding: [0x0b,0x9c,0xd6,0x11]
            evmhesmf %r14, %r22, %r19
# CHECK-BE: evmhesmfa 14, 22, 19            # encoding: [0x11,0xd6,0x9c,0x2b]
# CHECK-LE: evmhesmfa 14, 22, 19            # encoding: [0x2b,0x9c,0xd6,0x11]
            evmhesmfa %r14, %r22, %r19
# CHECK-BE: evmhesmfaaw 14, 22, 19          # encoding: [0x11,0xd6,0x9d,0x0b]
# CHECK-LE: evmhesmfaaw 14, 22, 19          # encoding: [0x0b,0x9d,0xd6,0x11]
            evmhesmfaaw %r14, %r22, %r19
# CHECK-BE: evmhesmfanw 14, 22, 19          # encoding: [0x11,0xd6,0x9d,0x8b]
# CHECK-LE: evmhesmfanw 14, 22, 19          # encoding: [0x8b,0x9d,0xd6,0x11]
            evmhesmfanw %r14, %r22, %r19
# CHECK-BE: evmhesmi 14, 22, 19             # encoding: [0x11,0xd6,0x9c,0x09]
# CHECK-LE: evmhesmi 14, 22, 19             # encoding: [0x09,0x9c,0xd6,0x11]
            evmhesmi %r14, %r22, %r19
# CHECK-BE: evmhesmia 14, 22, 19            # encoding: [0x11,0xd6,0x9c,0x29]
# CHECK-LE: evmhesmia 14, 22, 19            # encoding: [0x29,0x9c,0xd6,0x11]
            evmhesmia %r14, %r22, %r19
# CHECK-BE: evmhesmiaaw 14, 22, 19          # encoding: [0x11,0xd6,0x9d,0x09]
# CHECK-LE: evmhesmiaaw 14, 22, 19          # encoding: [0x09,0x9d,0xd6,0x11]
            evmhesmiaaw %r14, %r22, %r19
# CHECK-BE: evmhesmianw 14, 22, 19          # encoding: [0x11,0xd6,0x9d,0x89]
# CHECK-LE: evmhesmianw 14, 22, 19          # encoding: [0x89,0x9d,0xd6,0x11]
            evmhesmianw %r14, %r22, %r19
# CHECK-BE: evmhessf 14, 22, 19             # encoding: [0x11,0xd6,0x9c,0x03]
# CHECK-LE: evmhessf 14, 22, 19             # encoding: [0x03,0x9c,0xd6,0x11]
            evmhessf %r14, %r22, %r19
# CHECK-BE: evmhessfa 14, 22, 19            # encoding: [0x11,0xd6,0x9c,0x23]
# CHECK-LE: evmhessfa 14, 22, 19            # encoding: [0x23,0x9c,0xd6,0x11]
            evmhessfa %r14, %r22, %r19
# CHECK-BE: evmhessfaaw 14, 22, 19          # encoding: [0x11,0xd6,0x9d,0x03]
# CHECK-LE: evmhessfaaw 14, 22, 19          # encoding: [0x03,0x9d,0xd6,0x11]
            evmhessfaaw %r14, %r22, %r19
# CHECK-BE: evmhessfanw 14, 22, 19          # encoding: [0x11,0xd6,0x9d,0x83]
# CHECK-LE: evmhessfanw 14, 22, 19          # encoding: [0x83,0x9d,0xd6,0x11]
            evmhessfanw %r14, %r22, %r19
# CHECK-BE: evmhessiaaw 14, 22, 19          # encoding: [0x11,0xd6,0x9d,0x01]
# CHECK-LE: evmhessiaaw 14, 22, 19          # encoding: [0x01,0x9d,0xd6,0x11]
            evmhessiaaw %r14, %r22, %r19
# CHECK-BE: evmhessianw 14, 22, 19          # encoding: [0x11,0xd6,0x9d,0x81]
# CHECK-LE: evmhessianw 14, 22, 19          # encoding: [0x81,0x9d,0xd6,0x11]
            evmhessianw %r14, %r22, %r19
# CHECK-BE: evmheumi 14, 22, 19             # encoding: [0x11,0xd6,0x9c,0x08]
# CHECK-LE: evmheumi 14, 22, 19             # encoding: [0x08,0x9c,0xd6,0x11]
            evmheumi %r14, %r22, %r19
# CHECK-BE: evmheumia 14, 22, 19            # encoding: [0x11,0xd6,0x9c,0x28]
# CHECK-LE: evmheumia 14, 22, 19            # encoding: [0x28,0x9c,0xd6,0x11]
            evmheumia %r14, %r22, %r19
# CHECK-BE: evmheumiaaw 14, 22, 19          # encoding: [0x11,0xd6,0x9d,0x08]
# CHECK-LE: evmheumiaaw 14, 22, 19          # encoding: [0x08,0x9d,0xd6,0x11]
            evmheumiaaw %r14, %r22, %r19
# CHECK-BE: evmheumianw 14, 22, 19          # encoding: [0x11,0xd6,0x9d,0x88]
# CHECK-LE: evmheumianw 14, 22, 19          # encoding: [0x88,0x9d,0xd6,0x11]
            evmheumianw %r14, %r22, %r19
# CHECK-BE: evmheusiaaw 14, 22, 19          # encoding: [0x11,0xd6,0x9d,0x00]
# CHECK-LE: evmheusiaaw 14, 22, 19          # encoding: [0x00,0x9d,0xd6,0x11]
            evmheusiaaw %r14, %r22, %r19
# CHECK-BE: evmheusianw 14, 22, 19          # encoding: [0x11,0xd6,0x9d,0x80]
# CHECK-LE: evmheusianw 14, 22, 19          # encoding: [0x80,0x9d,0xd6,0x11]
            evmheusianw %r14, %r22, %r19
# CHECK-BE: evmhogsmfaa 14, 22, 19          # encoding: [0x11,0xd6,0x9d,0x2f]
# CHECK-LE: evmhogsmfaa 14, 22, 19          # encoding: [0x2f,0x9d,0xd6,0x11]
            evmhogsmfaa %r14, %r22, %r19
# CHECK-BE: evmhogsmfan 14, 22, 19          # encoding: [0x11,0xd6,0x9d,0xaf]
# CHECK-LE: evmhogsmfan 14, 22, 19          # encoding: [0xaf,0x9d,0xd6,0x11]
            evmhogsmfan %r14, %r22, %r19
# CHECK-BE: evmhogsmiaa 14, 22, 19          # encoding: [0x11,0xd6,0x9d,0x2d]
# CHECK-LE: evmhogsmiaa 14, 22, 19          # encoding: [0x2d,0x9d,0xd6,0x11]
            evmhogsmiaa %r14, %r22, %r19
# CHECK-BE: evmhogsmian 14, 22, 19          # encoding: [0x11,0xd6,0x9d,0xad]
# CHECK-LE: evmhogsmian 14, 22, 19          # encoding: [0xad,0x9d,0xd6,0x11]
            evmhogsmian %r14, %r22, %r19
# CHECK-BE: evmhogumiaa 14, 22, 19          # encoding: [0x11,0xd6,0x9d,0x2c]
# CHECK-LE: evmhogumiaa 14, 22, 19          # encoding: [0x2c,0x9d,0xd6,0x11]
            evmhogumiaa %r14, %r22, %r19
# CHECK-BE: evmhogumian 14, 22, 19          # encoding: [0x11,0xd6,0x9d,0xac]
# CHECK-LE: evmhogumian 14, 22, 19          # encoding: [0xac,0x9d,0xd6,0x11]
            evmhogumian %r14, %r22, %r19
# CHECK-BE: evmhosmf 14, 22, 19             # encoding: [0x11,0xd6,0x9c,0x0f]
# CHECK-LE: evmhosmf 14, 22, 19             # encoding: [0x0f,0x9c,0xd6,0x11]
            evmhosmf %r14, %r22, %r19
# CHECK-BE: evmhosmfa 14, 22, 19            # encoding: [0x11,0xd6,0x9c,0x2f]
# CHECK-LE: evmhosmfa 14, 22, 19            # encoding: [0x2f,0x9c,0xd6,0x11]
            evmhosmfa %r14, %r22, %r19
# CHECK-BE: evmhosmfaaw 14, 22, 19          # encoding: [0x11,0xd6,0x9d,0x0f]
# CHECK-LE: evmhosmfaaw 14, 22, 19          # encoding: [0x0f,0x9d,0xd6,0x11]
            evmhosmfaaw %r14, %r22, %r19
# CHECK-BE: evmhosmfanw 14, 22, 19          # encoding: [0x11,0xd6,0x9d,0x8f]
# CHECK-LE: evmhosmfanw 14, 22, 19          # encoding: [0x8f,0x9d,0xd6,0x11]
            evmhosmfanw %r14, %r22, %r19
# CHECK-BE: evmhosmi 14, 22, 19             # encoding: [0x11,0xd6,0x9c,0x0d]
# CHECK-LE: evmhosmi 14, 22, 19             # encoding: [0x0d,0x9c,0xd6,0x11]
            evmhosmi %r14, %r22, %r19
# CHECK-BE: evmhosmia 14, 22, 19            # encoding: [0x11,0xd6,0x9c,0x2d]
# CHECK-LE: evmhosmia 14, 22, 19            # encoding: [0x2d,0x9c,0xd6,0x11]
            evmhosmia %r14, %r22, %r19
# CHECK-BE: evmhosmiaaw 14, 22, 19          # encoding: [0x11,0xd6,0x9d,0x0d]
# CHECK-LE: evmhosmiaaw 14, 22, 19          # encoding: [0x0d,0x9d,0xd6,0x11]
            evmhosmiaaw %r14, %r22, %r19
# CHECK-BE: evmhosmianw 14, 22, 19          # encoding: [0x11,0xd6,0x9d,0x8d]
# CHECK-LE: evmhosmianw 14, 22, 19          # encoding: [0x8d,0x9d,0xd6,0x11]
            evmhosmianw %r14, %r22, %r19
# CHECK-BE: evmhossf 14, 22, 19             # encoding: [0x11,0xd6,0x9c,0x07]
# CHECK-LE: evmhossf 14, 22, 19             # encoding: [0x07,0x9c,0xd6,0x11]
            evmhossf %r14, %r22, %r19
# CHECK-BE: evmhossfa 14, 22, 19            # encoding: [0x11,0xd6,0x9c,0x27]
# CHECK-LE: evmhossfa 14, 22, 19            # encoding: [0x27,0x9c,0xd6,0x11]
            evmhossfa %r14, %r22, %r19
# CHECK-BE: evmhossfaaw 14, 22, 19          # encoding: [0x11,0xd6,0x9d,0x07]
# CHECK-LE: evmhossfaaw 14, 22, 19          # encoding: [0x07,0x9d,0xd6,0x11]
            evmhossfaaw %r14, %r22, %r19
# CHECK-BE: evmhossfanw 14, 22, 19          # encoding: [0x11,0xd6,0x9d,0x87]
# CHECK-LE: evmhossfanw 14, 22, 19          # encoding: [0x87,0x9d,0xd6,0x11]
            evmhossfanw %r14, %r22, %r19
# CHECK-BE: evmhossiaaw 14, 22, 19          # encoding: [0x11,0xd6,0x9d,0x05]
# CHECK-LE: evmhossiaaw 14, 22, 19          # encoding: [0x05,0x9d,0xd6,0x11]
            evmhossiaaw %r14, %r22, %r19
# CHECK-BE: evmhossianw 14, 22, 19          # encoding: [0x11,0xd6,0x9d,0x85]
# CHECK-LE: evmhossianw 14, 22, 19          # encoding: [0x85,0x9d,0xd6,0x11]
            evmhossianw %r14, %r22, %r19
# CHECK-BE: evmhoumi 14, 22, 19             # encoding: [0x11,0xd6,0x9c,0x0c]
# CHECK-LE: evmhoumi 14, 22, 19             # encoding: [0x0c,0x9c,0xd6,0x11]
            evmhoumi %r14, %r22, %r19
# CHECK-BE: evmhoumia 14, 22, 19            # encoding: [0x11,0xd6,0x9c,0x2c]
# CHECK-LE: evmhoumia 14, 22, 19            # encoding: [0x2c,0x9c,0xd6,0x11]
            evmhoumia %r14, %r22, %r19
# CHECK-BE: evmhoumiaaw 14, 22, 19          # encoding: [0x11,0xd6,0x9d,0x0c]
# CHECK-LE: evmhoumiaaw 14, 22, 19          # encoding: [0x0c,0x9d,0xd6,0x11]
            evmhoumiaaw %r14, %r22, %r19
# CHECK-BE: evmhoumianw 14, 22, 19          # encoding: [0x11,0xd6,0x9d,0x8c]
# CHECK-LE: evmhoumianw 14, 22, 19          # encoding: [0x8c,0x9d,0xd6,0x11]
            evmhoumianw %r14, %r22, %r19
# CHECK-BE: evmhousiaaw 14, 22, 19          # encoding: [0x11,0xd6,0x9d,0x04]
# CHECK-LE: evmhousiaaw 14, 22, 19          # encoding: [0x04,0x9d,0xd6,0x11]
            evmhousiaaw %r14, %r22, %r19
# CHECK-BE: evmhousianw 14, 22, 19          # encoding: [0x11,0xd6,0x9d,0x84]
# CHECK-LE: evmhousianw 14, 22, 19          # encoding: [0x84,0x9d,0xd6,0x11]
            evmhousianw %r14, %r22, %r19
# CHECK-BE: evmwhsmf 14, 22, 19             # encoding: [0x11,0xd6,0x9c,0x4f]
# CHECK-LE: evmwhsmf 14, 22, 19             # encoding: [0x4f,0x9c,0xd6,0x11]
            evmwhsmf %r14, %r22, %r19
# CHECK-BE: evmwhsmfa 14, 22, 19            # encoding: [0x11,0xd6,0x9c,0x6f]
# CHECK-LE: evmwhsmfa 14, 22, 19            # encoding: [0x6f,0x9c,0xd6,0x11]
            evmwhsmfa %r14, %r22, %r19
# CHECK-BE: evmwhsmi 14, 22, 19             # encoding: [0x11,0xd6,0x9c,0x4d]
# CHECK-LE: evmwhsmi 14, 22, 19             # encoding: [0x4d,0x9c,0xd6,0x11]
            evmwhsmi %r14, %r22, %r19
# CHECK-BE: evmwhsmia 14, 22, 19            # encoding: [0x11,0xd6,0x9c,0x6d]
# CHECK-LE: evmwhsmia 14, 22, 19            # encoding: [0x6d,0x9c,0xd6,0x11]
            evmwhsmia %r14, %r22, %r19
# CHECK-BE: evmwhssf 14, 22, 19             # encoding: [0x11,0xd6,0x9c,0x47]
# CHECK-LE: evmwhssf 14, 22, 19             # encoding: [0x47,0x9c,0xd6,0x11]
            evmwhssf %r14, %r22, %r19
# CHECK-BE: evmwhssfa 14, 22, 19            # encoding: [0x11,0xd6,0x9c,0x67]
# CHECK-LE: evmwhssfa 14, 22, 19            # encoding: [0x67,0x9c,0xd6,0x11]
            evmwhssfa %r14, %r22, %r19
# CHECK-BE: evmwhumi 14, 22, 19             # encoding: [0x11,0xd6,0x9c,0x4c]
# CHECK-LE: evmwhumi 14, 22, 19             # encoding: [0x4c,0x9c,0xd6,0x11]
            evmwhumi %r14, %r22, %r19
# CHECK-BE: evmwhumia 14, 22, 19            # encoding: [0x11,0xd6,0x9c,0x6c]
# CHECK-LE: evmwhumia 14, 22, 19            # encoding: [0x6c,0x9c,0xd6,0x11]
            evmwhumia %r14, %r22, %r19
# CHECK-BE: evmwlsmiaaw 14, 22, 19          # encoding: [0x11,0xd6,0x9d,0x49]
# CHECK-LE: evmwlsmiaaw 14, 22, 19          # encoding: [0x49,0x9d,0xd6,0x11]
            evmwlsmiaaw %r14, %r22, %r19
# CHECK-BE: evmwlsmianw 14, 22, 19          # encoding: [0x11,0xd6,0x9d,0xc9]
# CHECK-LE: evmwlsmianw 14, 22, 19          # encoding: [0xc9,0x9d,0xd6,0x11]
            evmwlsmianw %r14, %r22, %r19
# CHECK-BE: evmwlssiaaw 14, 22, 19          # encoding: [0x11,0xd6,0x9d,0x41]
# CHECK-LE: evmwlssiaaw 14, 22, 19          # encoding: [0x41,0x9d,0xd6,0x11]
            evmwlssiaaw %r14, %r22, %r19
# CHECK-BE: evmwlssianw 14, 22, 19          # encoding: [0x11,0xd6,0x9d,0xc1]
# CHECK-LE: evmwlssianw 14, 22, 19          # encoding: [0xc1,0x9d,0xd6,0x11]
            evmwlssianw %r14, %r22, %r19
# CHECK-BE: evmwlumi 14, 22, 19             # encoding: [0x11,0xd6,0x9c,0x48]
# CHECK-LE: evmwlumi 14, 22, 19             # encoding: [0x48,0x9c,0xd6,0x11]
            evmwlumi %r14, %r22, %r19
# CHECK-BE: evmwlumia 14, 22, 19            # encoding: [0x11,0xd6,0x9c,0x68]
# CHECK-LE: evmwlumia 14, 22, 19            # encoding: [0x68,0x9c,0xd6,0x11]
            evmwlumia %r14, %r22, %r19
# CHECK-BE: evmwlumiaaw 14, 22, 19          # encoding: [0x11,0xd6,0x9d,0x48]
# CHECK-LE: evmwlumiaaw 14, 22, 19          # encoding: [0x48,0x9d,0xd6,0x11]
            evmwlumiaaw %r14, %r22, %r19
# CHECK-BE: evmwlumianw 14, 22, 19          # encoding: [0x11,0xd6,0x9d,0xc8]
# CHECK-LE: evmwlumianw 14, 22, 19          # encoding: [0xc8,0x9d,0xd6,0x11]
            evmwlumianw %r14, %r22, %r19
# CHECK-BE: evmwlusiaaw 14, 22, 19          # encoding: [0x11,0xd6,0x9d,0x40]
# CHECK-LE: evmwlusiaaw 14, 22, 19          # encoding: [0x40,0x9d,0xd6,0x11]
            evmwlusiaaw %r14, %r22, %r19
# CHECK-BE: evmwlusianw 14, 22, 19          # encoding: [0x11,0xd6,0x9d,0xc0]
# CHECK-LE: evmwlusianw 14, 22, 19          # encoding: [0xc0,0x9d,0xd6,0x11]
            evmwlusianw %r14, %r22, %r19
# CHECK-BE: evmwsmf 14, 22, 19              # encoding: [0x11,0xd6,0x9c,0x5b]
# CHECK-LE: evmwsmf 14, 22, 19              # encoding: [0x5b,0x9c,0xd6,0x11]
            evmwsmf %r14, %r22, %r19
# CHECK-BE: evmwsmfa 14, 22, 19             # encoding: [0x11,0xd6,0x9c,0x7b]
# CHECK-LE: evmwsmfa 14, 22, 19             # encoding: [0x7b,0x9c,0xd6,0x11]
            evmwsmfa %r14, %r22, %r19
# CHECK-BE: evmwsmfaa 14, 22, 19            # encoding: [0x11,0xd6,0x9d,0x5b]
# CHECK-LE: evmwsmfaa 14, 22, 19            # encoding: [0x5b,0x9d,0xd6,0x11]
            evmwsmfaa %r14, %r22, %r19
# CHECK-BE: evmwsmfan 14, 22, 19            # encoding: [0x11,0xd6,0x9d,0xdb]
# CHECK-LE: evmwsmfan 14, 22, 19            # encoding: [0xdb,0x9d,0xd6,0x11]
            evmwsmfan %r14, %r22, %r19
# CHECK-BE: evmwsmi 14, 22, 19              # encoding: [0x11,0xd6,0x9c,0x59]
# CHECK-LE: evmwsmi 14, 22, 19              # encoding: [0x59,0x9c,0xd6,0x11]
            evmwsmi %r14, %r22, %r19
# CHECK-BE: evmwsmia 14, 22, 19             # encoding: [0x11,0xd6,0x9c,0x79]
# CHECK-LE: evmwsmia 14, 22, 19             # encoding: [0x79,0x9c,0xd6,0x11]
            evmwsmia %r14, %r22, %r19
# CHECK-BE: evmwsmiaa 14, 22, 19            # encoding: [0x11,0xd6,0x9d,0x59]
# CHECK-LE: evmwsmiaa 14, 22, 19            # encoding: [0x59,0x9d,0xd6,0x11]
            evmwsmiaa %r14, %r22, %r19
# CHECK-BE: evmwsmian 14, 22, 19            # encoding: [0x11,0xd6,0x9d,0xd9]
# CHECK-LE: evmwsmian 14, 22, 19            # encoding: [0xd9,0x9d,0xd6,0x11]
            evmwsmian %r14, %r22, %r19
# CHECK-BE: evmwssf 14, 22, 19              # encoding: [0x11,0xd6,0x9c,0x53]
# CHECK-LE: evmwssf 14, 22, 19              # encoding: [0x53,0x9c,0xd6,0x11]
            evmwssf %r14, %r22, %r19
# CHECK-BE: evmwssfa 14, 22, 19             # encoding: [0x11,0xd6,0x9c,0x73]
# CHECK-LE: evmwssfa 14, 22, 19             # encoding: [0x73,0x9c,0xd6,0x11]
            evmwssfa %r14, %r22, %r19
# CHECK-BE: evmwssfaa 14, 22, 19            # encoding: [0x11,0xd6,0x9d,0x53]
# CHECK-LE: evmwssfaa 14, 22, 19            # encoding: [0x53,0x9d,0xd6,0x11]
            evmwssfaa %r14, %r22, %r19
# CHECK-BE: evmwssfan 14, 22, 19            # encoding: [0x11,0xd6,0x9d,0xd3]
# CHECK-LE: evmwssfan 14, 22, 19            # encoding: [0xd3,0x9d,0xd6,0x11]
            evmwssfan %r14, %r22, %r19
# CHECK-BE: evmwumi 14, 22, 19              # encoding: [0x11,0xd6,0x9c,0x58]
# CHECK-LE: evmwumi 14, 22, 19              # encoding: [0x58,0x9c,0xd6,0x11]
            evmwumi %r14, %r22, %r19
# CHECK-BE: evmwumia 14, 22, 19             # encoding: [0x11,0xd6,0x9c,0x78]
# CHECK-LE: evmwumia 14, 22, 19             # encoding: [0x78,0x9c,0xd6,0x11]
            evmwumia %r14, %r22, %r19
# CHECK-BE: evmwumiaa 14, 22, 19            # encoding: [0x11,0xd6,0x9d,0x58]
# CHECK-LE: evmwumiaa 14, 22, 19            # encoding: [0x58,0x9d,0xd6,0x11]
            evmwumiaa %r14, %r22, %r19
# CHECK-BE: evmwumian 14, 22, 19            # encoding: [0x11,0xd6,0x9d,0xd8]
# CHECK-LE: evmwumian 14, 22, 19            # encoding: [0xd8,0x9d,0xd6,0x11]
            evmwumian %r14, %r22, %r19
# CHECK-BE: evnand 14, 22, 19               # encoding: [0x11,0xd6,0x9a,0x1e]
# CHECK-LE: evnand 14, 22, 19               # encoding: [0x1e,0x9a,0xd6,0x11]
            evnand %r14, %r22, %r19
# CHECK-BE: evneg 14, 22                    # encoding: [0x11,0xd6,0x02,0x09]
# CHECK-LE: evneg 14, 22                    # encoding: [0x09,0x02,0xd6,0x11]
            evneg %r14, %r22
# CHECK-BE: evnor 14, 22, 19                # encoding: [0x11,0xd6,0x9a,0x18]
# CHECK-LE: evnor 14, 22, 19                # encoding: [0x18,0x9a,0xd6,0x11]
            evnor %r14, %r22, %r19
# CHECK-BE: evor 14, 22, 19                 # encoding: [0x11,0xd6,0x9a,0x17]
# CHECK-LE: evor 14, 22, 19                 # encoding: [0x17,0x9a,0xd6,0x11]
            evor %r14, %r22, %r19
# CHECK-BE: evorc 14, 22, 19                # encoding: [0x11,0xd6,0x9a,0x1b]
# CHECK-LE: evorc 14, 22, 19                # encoding: [0x1b,0x9a,0xd6,0x11]
            evorc %r14, %r22, %r19
# CHECK-BE: evrlwi 14, 29, 19               # encoding: [0x11,0xdd,0x9a,0x2a]
# CHECK-LE: evrlwi 14, 29, 19               # encoding: [0x2a,0x9a,0xdd,0x11]
            evrlwi %r14, 29, %r19
# CHECK-BE: evrlw 14, 22, 19                # encoding: [0x11,0xd6,0x9a,0x28]
# CHECK-LE: evrlw 14, 22, 19                # encoding: [0x28,0x9a,0xd6,0x11]
            evrlw %r14, %r22, %r19
# CHECK-BE: evrndw 14, 22                   # encoding: [0x11,0xd6,0x02,0x0c]
# CHECK-LE: evrndw 14, 22                   # encoding: [0x0c,0x02,0xd6,0x11]
            evrndw %r14, %r22
# CHECK-BE: evslwi 14, 29, 19               # encoding: [0x11,0xdd,0x9a,0x26]
# CHECK-LE: evslwi 14, 29, 19               # encoding: [0x26,0x9a,0xdd,0x11]
            evslwi %r14, 29, %r19
# CHECK-BE: evslw 14, 22, 19                # encoding: [0x11,0xd6,0x9a,0x24]
# CHECK-LE: evslw 14, 22, 19                # encoding: [0x24,0x9a,0xd6,0x11]
            evslw %r14, %r22, %r19
# CHECK-BE: evsplatfi 14, -13               # encoding: [0x11,0xd3,0x02,0x2b]
# CHECK-LE: evsplatfi 14, -13               # encoding: [0x2b,0x02,0xd3,0x11]
            evsplatfi %r14, -13
# CHECK-BE: evsplati 14, -13                # encoding: [0x11,0xd3,0x02,0x29]
# CHECK-LE: evsplati 14, -13                # encoding: [0x29,0x02,0xd3,0x11]
            evsplati %r14, -13
# CHECK-BE: evsrwis 14, 29, 19              # encoding: [0x11,0xdd,0x9a,0x23]
# CHECK-LE: evsrwis 14, 29, 19              # encoding: [0x23,0x9a,0xdd,0x11]
            evsrwis %r14, 29, %r19
# CHECK-BE: evsrwiu 14, 29, 19              # encoding: [0x11,0xdd,0x9a,0x22]
# CHECK-LE: evsrwiu 14, 29, 19              # encoding: [0x22,0x9a,0xdd,0x11]
            evsrwiu %r14, 29, %r19
# CHECK-BE: evsrws 14, 22, 19               # encoding: [0x11,0xd6,0x9a,0x21]
# CHECK-LE: evsrws 14, 22, 19               # encoding: [0x21,0x9a,0xd6,0x11]
            evsrws %r14, %r22, %r19
# CHECK-BE: evsrwu 14, 22, 19               # encoding: [0x11,0xd6,0x9a,0x20]
# CHECK-LE: evsrwu 14, 22, 19               # encoding: [0x20,0x9a,0xd6,0x11]
            evsrwu %r14, %r22, %r19
# CHECK-BE: evstddx 14, 22, 19              # encoding: [0x11,0xd6,0x9b,0x20]
# CHECK-LE: evstddx 14, 22, 19              # encoding: [0x20,0x9b,0xd6,0x11]
            evstddx %r14, %r22, %r19
# CHECK-BE: evstdhx 14, 22, 19              # encoding: [0x11,0xd6,0x9b,0x24]
# CHECK-LE: evstdhx 14, 22, 19              # encoding: [0x24,0x9b,0xd6,0x11]
            evstdhx %r14, %r22, %r19
# CHECK-BE: evstdwx 14, 22, 19              # encoding: [0x11,0xd6,0x9b,0x22]
# CHECK-LE: evstdwx 14, 22, 19              # encoding: [0x22,0x9b,0xd6,0x11]
            evstdwx %r14, %r22, %r19
# CHECK-BE: evstwhex 14, 22, 19             # encoding: [0x11,0xd6,0x9b,0x30]
# CHECK-LE: evstwhex 14, 22, 19             # encoding: [0x30,0x9b,0xd6,0x11]
            evstwhex %r14, %r22, %r19
# CHECK-BE: evstwhox 14, 22, 19             # encoding: [0x11,0xd6,0x9b,0x34]
# CHECK-LE: evstwhox 14, 22, 19             # encoding: [0x34,0x9b,0xd6,0x11]
            evstwhox %r14, %r22, %r19
# CHECK-BE: evstwwex 14, 22, 19             # encoding: [0x11,0xd6,0x9b,0x38]
# CHECK-LE: evstwwex 14, 22, 19             # encoding: [0x38,0x9b,0xd6,0x11]
            evstwwex %r14, %r22, %r19
# CHECK-BE: evstwwox 14, 22, 19             # encoding: [0x11,0xd6,0x9b,0x3c]
# CHECK-LE: evstwwox 14, 22, 19             # encoding: [0x3c,0x9b,0xd6,0x11]
            evstwwox %r14, %r22, %r19
# CHECK-BE: evsubfssiaaw 14, 22             # encoding: [0x11,0xd6,0x04,0xc3]
# CHECK-LE: evsubfssiaaw 14, 22             # encoding: [0xc3,0x04,0xd6,0x11]
            evsubfssiaaw %r14, %r22
# CHECK-BE: evsubfsmiaaw 14, 22             # encoding: [0x11,0xd6,0x04,0xcb]
# CHECK-LE: evsubfsmiaaw 14, 22             # encoding: [0xcb,0x04,0xd6,0x11]
            evsubfsmiaaw %r14, %r22
# CHECK-BE: evsubfumiaaw 14, 22             # encoding: [0x11,0xd6,0x04,0xca]
# CHECK-LE: evsubfumiaaw 14, 22             # encoding: [0xca,0x04,0xd6,0x11]
            evsubfumiaaw %r14, %r22
# CHECK-BE: evsubfusiaaw 14, 22             # encoding: [0x11,0xd6,0x04,0xc2]
# CHECK-LE: evsubfusiaaw 14, 22             # encoding: [0xc2,0x04,0xd6,0x11]
            evsubfusiaaw %r14, %r22
# CHECK-BE: evsubfw 14, 22, 19              # encoding: [0x11,0xd6,0x9a,0x04]
# CHECK-LE: evsubfw 14, 22, 19              # encoding: [0x04,0x9a,0xd6,0x11]
            evsubfw %r14, %r22, %r19
# CHECK-BE: evsubifw 14, 29, 19             # encoding: [0x11,0xdd,0x9a,0x06]
# CHECK-LE: evsubifw 14, 29, 19             # encoding: [0x06,0x9a,0xdd,0x11]
            evsubifw %r14, 29, %r19
# CHECK-BE: evxor 14, 22, 19                # encoding: [0x11,0xd6,0x9a,0x16]
# CHECK-LE: evxor 14, 22, 19                # encoding: [0x16,0x9a,0xd6,0x11]
            evxor %r14, %r22, %r19

# CHECK-BE: evldd 14, 0(27)                 # encoding: [0x11,0xdb,0x03,0x01]
# CHECK-LE: evldd 14, 0(27)                 # encoding: [0x01,0x03,0xdb,0x11]
            evldd %r14, 0(%r27)
# CHECK-BE: evldd 14, 248(27)               # encoding: [0x11,0xdb,0xfb,0x01]
# CHECK-LE: evldd 14, 248(27)               # encoding: [0x01,0xfb,0xdb,0x11]
            evldd %r14, 248(%r27)
# CHECK-BE: evldd 14, 248(9)                # encoding: [0x11,0xc9,0xfb,0x01]
# CHECK-LE: evldd 14, 248(9)                # encoding: [0x01,0xfb,0xc9,0x11]
            evldd %r14, 248(%r9)
# CHECK-BE: evldw 14, 0(27)                 # encoding: [0x11,0xdb,0x03,0x03]
# CHECK-LE: evldw 14, 0(27)                 # encoding: [0x03,0x03,0xdb,0x11]
            evldw %r14, 0(%r27)
# CHECK-BE: evldw 14, 248(27)               # encoding: [0x11,0xdb,0xfb,0x03]
# CHECK-LE: evldw 14, 248(27)               # encoding: [0x03,0xfb,0xdb,0x11]
            evldw %r14, 248(%r27)
# CHECK-BE: evldw 14, 248(9)                # encoding: [0x11,0xc9,0xfb,0x03]
# CHECK-LE: evldw 14, 248(9)                # encoding: [0x03,0xfb,0xc9,0x11]
            evldw %r14, 248(%r9)
# CHECK-BE: evldh 14, 0(27)                 # encoding: [0x11,0xdb,0x03,0x05]
# CHECK-LE: evldh 14, 0(27)                 # encoding: [0x05,0x03,0xdb,0x11]
            evldh %r14, 0(%r27)
# CHECK-BE: evldh 14, 248(27)               # encoding: [0x11,0xdb,0xfb,0x05]
# CHECK-LE: evldh 14, 248(27)               # encoding: [0x05,0xfb,0xdb,0x11]
            evldh %r14, 248(%r27)
# CHECK-BE: evldh 14, 248(9)                # encoding: [0x11,0xc9,0xfb,0x05]
# CHECK-LE: evldh 14, 248(9)                # encoding: [0x05,0xfb,0xc9,0x11]
            evldh %r14, 248(%r9)
# CHECK-BE: evlhhesplat 14, 0(27)           # encoding: [0x11,0xdb,0x03,0x09]
# CHECK-LE: evlhhesplat 14, 0(27)           # encoding: [0x09,0x03,0xdb,0x11]
            evlhhesplat %r14, 0(%r27)
# CHECK-BE: evlhhousplat 14, 0(27)          # encoding: [0x11,0xdb,0x03,0x0d]
# CHECK-LE: evlhhousplat 14, 0(27)          # encoding: [0x0d,0x03,0xdb,0x11]
            evlhhousplat %r14, 0(%r27)
# CHECK-BE: evlhhousplat 14, 62(27)         # encoding: [0x11,0xdb,0xfb,0x0d]
# CHECK-LE: evlhhousplat 14, 62(27)         # encoding: [0x0d,0xfb,0xdb,0x11]
            evlhhousplat %r14, 62(%r27)
# CHECK-BE: evlhhousplat 14, 62(9)          # encoding: [0x11,0xc9,0xfb,0x0d]
# CHECK-LE: evlhhousplat 14, 62(9)          # encoding: [0x0d,0xfb,0xc9,0x11]
            evlhhousplat %r14, 62(%r9)
# CHECK-BE: evlhhossplat 14, 0(27)          # encoding: [0x11,0xdb,0x03,0x0f]
# CHECK-LE: evlhhossplat 14, 0(27)          # encoding: [0x0f,0x03,0xdb,0x11]
            evlhhossplat %r14, 0(%r27)
# CHECK-BE: evlhhossplat 14, 62(27)         # encoding: [0x11,0xdb,0xfb,0x0f]
# CHECK-LE: evlhhossplat 14, 62(27)         # encoding: [0x0f,0xfb,0xdb,0x11]
            evlhhossplat %r14, 62(%r27)
# CHECK-BE: evlhhossplat 14, 62(9)          # encoding: [0x11,0xc9,0xfb,0x0f]
# CHECK-LE: evlhhossplat 14, 62(9)          # encoding: [0x0f,0xfb,0xc9,0x11]
            evlhhossplat %r14, 62(%r9)
# CHECK-BE: evlwhe 14, 0(27)                # encoding: [0x11,0xdb,0x03,0x11]
# CHECK-LE: evlwhe 14, 0(27)                # encoding: [0x11,0x03,0xdb,0x11]
            evlwhe %r14, 0(%r27)
# CHECK-BE: evlwhe 14, 124(27)              # encoding: [0x11,0xdb,0xfb,0x11]
# CHECK-LE: evlwhe 14, 124(27)              # encoding: [0x11,0xfb,0xdb,0x11]
            evlwhe %r14, 124(%r27)
# CHECK-BE: evlwhe 14, 124(9)               # encoding: [0x11,0xc9,0xfb,0x11]
# CHECK-LE: evlwhe 14, 124(9)               # encoding: [0x11,0xfb,0xc9,0x11]
            evlwhe %r14, 124(%r9)
# CHECK-BE: evlwhou 14, 0(27)               # encoding: [0x11,0xdb,0x03,0x15]
# CHECK-LE: evlwhou 14, 0(27)               # encoding: [0x15,0x03,0xdb,0x11]
            evlwhou %r14, 0(%r27)
# CHECK-BE: evlwhou 14, 124(27)             # encoding: [0x11,0xdb,0xfb,0x15]
# CHECK-LE: evlwhou 14, 124(27)             # encoding: [0x15,0xfb,0xdb,0x11]
            evlwhou %r14, 124(%r27)
# CHECK-BE: evlwhou 14, 124(9)              # encoding: [0x11,0xc9,0xfb,0x15]
# CHECK-LE: evlwhou 14, 124(9)              # encoding: [0x15,0xfb,0xc9,0x11]
            evlwhou %r14, 124(%r9)
# CHECK-BE: evlwhos 14, 0(27)               # encoding: [0x11,0xdb,0x03,0x17]
# CHECK-LE: evlwhos 14, 0(27)               # encoding: [0x17,0x03,0xdb,0x11]
            evlwhos %r14, 0(%r27)
# CHECK-BE: evlwhos 14, 124(27)             # encoding: [0x11,0xdb,0xfb,0x17]
# CHECK-LE: evlwhos 14, 124(27)             # encoding: [0x17,0xfb,0xdb,0x11]
            evlwhos %r14, 124(%r27)
# CHECK-BE: evlwhos 14, 124(9)              # encoding: [0x11,0xc9,0xfb,0x17]
# CHECK-LE: evlwhos 14, 124(9)              # encoding: [0x17,0xfb,0xc9,0x11]
            evlwhos %r14, 124(%r9)
# CHECK-BE: evlwwsplat 14, 0(27)            # encoding: [0x11,0xdb,0x03,0x19]
# CHECK-LE: evlwwsplat 14, 0(27)            # encoding: [0x19,0x03,0xdb,0x11]
            evlwwsplat %r14, 0(%r27)
# CHECK-BE: evlwwsplat 14, 124(27)          # encoding: [0x11,0xdb,0xfb,0x19]
# CHECK-LE: evlwwsplat 14, 124(27)          # encoding: [0x19,0xfb,0xdb,0x11]
            evlwwsplat %r14, 124(%r27)
# CHECK-BE: evlwwsplat 14, 124(9)           # encoding: [0x11,0xc9,0xfb,0x19]
# CHECK-LE: evlwwsplat 14, 124(9)           # encoding: [0x19,0xfb,0xc9,0x11]
            evlwwsplat %r14, 124(%r9)
# CHECK-BE: evlwhsplat 14, 0(27)            # encoding: [0x11,0xdb,0x03,0x1d]
# CHECK-LE: evlwhsplat 14, 0(27)            # encoding: [0x1d,0x03,0xdb,0x11]
            evlwhsplat %r14, 0(%r27)
# CHECK-BE: evlwhsplat 14, 124(27)          # encoding: [0x11,0xdb,0xfb,0x1d]
# CHECK-LE: evlwhsplat 14, 124(27)          # encoding: [0x1d,0xfb,0xdb,0x11]
            evlwhsplat %r14, 124(%r27)
# CHECK-BE: evlwhsplat 14, 124(9)           # encoding: [0x11,0xc9,0xfb,0x1d]
# CHECK-LE: evlwhsplat 14, 124(9)           # encoding: [0x1d,0xfb,0xc9,0x11]
            evlwhsplat %r14, 124(%r9)
# CHECK-BE: evstdd 14, 0(27)                # encoding: [0x11,0xdb,0x03,0x21]
# CHECK-LE: evstdd 14, 0(27)                # encoding: [0x21,0x03,0xdb,0x11]
            evstdd %r14, 0(%r27)
# CHECK-BE: evstdd 14, 248(27)              # encoding: [0x11,0xdb,0xfb,0x21]
# CHECK-LE: evstdd 14, 248(27)              # encoding: [0x21,0xfb,0xdb,0x11]
            evstdd %r14, 248(%r27)
# CHECK-BE: evstdd 14, 248(9)               # encoding: [0x11,0xc9,0xfb,0x21]
# CHECK-LE: evstdd 14, 248(9)               # encoding: [0x21,0xfb,0xc9,0x11]
            evstdd %r14, 248(%r9)
# CHECK-BE: evstdh 14, 0(27)                # encoding: [0x11,0xdb,0x03,0x25]
# CHECK-LE: evstdh 14, 0(27)                # encoding: [0x25,0x03,0xdb,0x11]
            evstdh %r14, 0(%r27)
# CHECK-BE: evstdh 14, 248(27)              # encoding: [0x11,0xdb,0xfb,0x25]
# CHECK-LE: evstdh 14, 248(27)              # encoding: [0x25,0xfb,0xdb,0x11]
            evstdh %r14, 248(%r27)
# CHECK-BE: evstdh 14, 248(9)               # encoding: [0x11,0xc9,0xfb,0x25]
# CHECK-LE: evstdh 14, 248(9)               # encoding: [0x25,0xfb,0xc9,0x11]
            evstdh %r14, 248(%r9)
# CHECK-BE: evstdw 14, 0(27)                # encoding: [0x11,0xdb,0x03,0x23]
# CHECK-LE: evstdw 14, 0(27)                # encoding: [0x23,0x03,0xdb,0x11]
            evstdw %r14, 0(%r27)
# CHECK-BE: evstdw 14, 248(27)              # encoding: [0x11,0xdb,0xfb,0x23]
# CHECK-LE: evstdw 14, 248(27)              # encoding: [0x23,0xfb,0xdb,0x11]
            evstdw %r14, 248(%r27)
# CHECK-BE: evstdw 14, 248(9)               # encoding: [0x11,0xc9,0xfb,0x23]
# CHECK-LE: evstdw 14, 248(9)               # encoding: [0x23,0xfb,0xc9,0x11]
            evstdw %r14, 248(%r9)
# CHECK-BE: evstwhe 14, 0(27)               # encoding: [0x11,0xdb,0x03,0x31]
# CHECK-LE: evstwhe 14, 0(27)               # encoding: [0x31,0x03,0xdb,0x11]
            evstwhe %r14, 0(%r27)
# CHECK-BE: evstwhe 14, 124(27)             # encoding: [0x11,0xdb,0xfb,0x31]
# CHECK-LE: evstwhe 14, 124(27)             # encoding: [0x31,0xfb,0xdb,0x11]
            evstwhe %r14, 124(%r27)
# CHECK-BE: evstwhe 14, 124(9)              # encoding: [0x11,0xc9,0xfb,0x31]
# CHECK-LE: evstwhe 14, 124(9)              # encoding: [0x31,0xfb,0xc9,0x11]
            evstwhe %r14, 124(%r9)
# CHECK-BE: evstwho 14, 0(27)               # encoding: [0x11,0xdb,0x03,0x35]
# CHECK-LE: evstwho 14, 0(27)               # encoding: [0x35,0x03,0xdb,0x11]
            evstwho %r14, 0(%r27)
# CHECK-BE: evstwho 14, 124(27)             # encoding: [0x11,0xdb,0xfb,0x35]
# CHECK-LE: evstwho 14, 124(27)             # encoding: [0x35,0xfb,0xdb,0x11]
            evstwho %r14, 124(%r27)
# CHECK-BE: evstwho 14, 124(9)              # encoding: [0x11,0xc9,0xfb,0x35]
# CHECK-LE: evstwho 14, 124(9)              # encoding: [0x35,0xfb,0xc9,0x11]
            evstwho %r14, 124(%r9)
# CHECK-BE: evstwwe 14, 0(27)               # encoding: [0x11,0xdb,0x03,0x39]
# CHECK-LE: evstwwe 14, 0(27)               # encoding: [0x39,0x03,0xdb,0x11]
            evstwwe %r14, 0(%r27)
# CHECK-BE: evstwwe 14, 124(27)             # encoding: [0x11,0xdb,0xfb,0x39]
# CHECK-LE: evstwwe 14, 124(27)             # encoding: [0x39,0xfb,0xdb,0x11]
            evstwwe %r14, 124(%r27)
# CHECK-BE: evstwwe 14, 124(9)              # encoding: [0x11,0xc9,0xfb,0x39]
# CHECK-LE: evstwwe 14, 124(9)              # encoding: [0x39,0xfb,0xc9,0x11]
            evstwwe %r14, 124(%r9)
# CHECK-BE: evstwwo 14, 0(27)               # encoding: [0x11,0xdb,0x03,0x3d]
# CHECK-LE: evstwwo 14, 0(27)               # encoding: [0x3d,0x03,0xdb,0x11]
            evstwwo %r14, 0(%r27)
# CHECK-BE: evstwwo 14, 124(27)             # encoding: [0x11,0xdb,0xfb,0x3d]
# CHECK-LE: evstwwo 14, 124(27)             # encoding: [0x3d,0xfb,0xdb,0x11]
            evstwwo %r14, 124(%r27)
# CHECK-BE: evstwwo 14, 124(9)              # encoding: [0x11,0xc9,0xfb,0x3d]
# CHECK-LE: evstwwo 14, 124(9)              # encoding: [0x3d,0xfb,0xc9,0x11]
            evstwwo %r14, 124(%r9)

# CHECK-BE: efdabs 3, 4                     # encoding: [0x10,0x64,0x02,0xe4]
# CHECK-LE: efdabs 3, 4                     # encoding: [0xe4,0x02,0x64,0x10]
            efdabs %r3, %r4
# CHECK-BE: efdadd 3, 4, 5                  # encoding: [0x10,0x64,0x2a,0xe0]
# CHECK-LE: efdadd 3, 4, 5                  # encoding: [0xe0,0x2a,0x64,0x10]
            efdadd %r3, %r4, %r5
# CHECK-BE: efdcfs 3, 4                     # encoding: [0x10,0x60,0x22,0xef]
# CHECK-LE: efdcfs 3, 4                     # encoding: [0xef,0x22,0x60,0x10]
            efdcfs %r3, %r4
# CHECK-BE: efdcfsf 5, 6                    # encoding: [0x10,0xa0,0x32,0xf3]
# CHECK-LE: efdcfsf 5, 6                    # encoding: [0xf3,0x32,0xa0,0x10]
            efdcfsf %r5, %r6
# CHECK-BE: efdcfsi 5, 6                    # encoding: [0x10,0xa0,0x32,0xf1]
# CHECK-LE: efdcfsi 5, 6                    # encoding: [0xf1,0x32,0xa0,0x10]
            efdcfsi %r5, %r6
# CHECK-BE: efdcfsid 10, 14                 # encoding: [0x11,0x40,0x72,0xe3]
# CHECK-LE: efdcfsid 10, 14                 # encoding: [0xe3,0x72,0x40,0x11]
            efdcfsid %r10, %r14
# CHECK-BE: efdcfuf 5, 8                    # encoding: [0x10,0xa0,0x42,0xf2]
# CHECK-LE: efdcfuf 5, 8                    # encoding: [0xf2,0x42,0xa0,0x10]
            efdcfuf %r5, %r8
# CHECK-BE: efdcfui 6, 9                    # encoding: [0x10,0xc0,0x4a,0xf0]
# CHECK-LE: efdcfui 6, 9                    # encoding: [0xf0,0x4a,0xc0,0x10]
            efdcfui %r6, %r9
# CHECK-BE: efdcfuid 7, 10                  # encoding: [0x10,0xe0,0x52,0xe2]
# CHECK-LE: efdcfuid 7, 10                  # encoding: [0xe2,0x52,0xe0,0x10]
            efdcfuid %r7, %r10
# CHECK-BE: efdcmpeq 3, 3, 8                # encoding: [0x11,0x83,0x42,0xee]
# CHECK-LE: efdcmpeq 3, 3, 8                # encoding: [0xee,0x42,0x83,0x11]
            efdcmpeq %cr3, %r3, %r8
# CHECK-BE: efdcmpgt 4, 7, 3                # encoding: [0x12,0x07,0x1a,0xec]
# CHECK-LE: efdcmpgt 4, 7, 3                # encoding: [0xec,0x1a,0x07,0x12]
            efdcmpgt %cr4, %r7, %r3
# CHECK-BE: efdcmplt 2, 3, 4                # encoding: [0x11,0x03,0x22,0xed]
# CHECK-LE: efdcmplt 2, 3, 4                # encoding: [0xed,0x22,0x03,0x11]
            efdcmplt %cr2, %r3, %r4
# CHECK-BE: efdctsf 5, 3                    # encoding: [0x10,0xa0,0x1a,0xf7]
# CHECK-LE: efdctsf 5, 3                    # encoding: [0xf7,0x1a,0xa0,0x10]
            efdctsf %r5, %r3
# CHECK-BE: efdctsi 6, 4                    # encoding: [0x10,0xc0,0x22,0xf5]
# CHECK-LE: efdctsi 6, 4                    # encoding: [0xf5,0x22,0xc0,0x10]
            efdctsi %r6, %r4
# CHECK-BE: efdctsidz 3, 4                  # encoding: [0x10,0x60,0x22,0xeb]
# CHECK-LE: efdctsidz 3, 4                  # encoding: [0xeb,0x22,0x60,0x10]
            efdctsidz %r3, %r4
# CHECK-BE: efdctsiz 3, 4                   # encoding: [0x10,0x60,0x22,0xfa]
# CHECK-LE: efdctsiz 3, 4                   # encoding: [0xfa,0x22,0x60,0x10]
            efdctsiz %r3, %r4
# CHECK-BE: efdctuf 5, 8                    # encoding: [0x10,0xa0,0x42,0xf6]
# CHECK-LE: efdctuf 5, 8                    # encoding: [0xf6,0x42,0xa0,0x10]
            efdctuf %r5, %r8
# CHECK-BE: efdctui 9, 10                   # encoding: [0x11,0x20,0x52,0xf4]
# CHECK-LE: efdctui 9, 10                   # encoding: [0xf4,0x52,0x20,0x11]
            efdctui %r9, %r10
# CHECK-BE: efdctuidz 3, 8                  # encoding: [0x10,0x60,0x42,0xea]
# CHECK-LE: efdctuidz 3, 8                  # encoding: [0xea,0x42,0x60,0x10]
            efdctuidz %r3, %r8
# CHECK-BE: efdctuiz 5, 17                  # encoding: [0x10,0xa0,0x8a,0xf8]
# CHECK-LE: efdctuiz 5, 17                  # encoding: [0xf8,0x8a,0xa0,0x10]
            efdctuiz %r5, %r17
# CHECK-BE: efddiv 3, 4, 5                  # encoding: [0x10,0x64,0x2a,0xe9]
# CHECK-LE: efddiv 3, 4, 5                  # encoding: [0xe9,0x2a,0x64,0x10]
            efddiv %r3, %r4, %r5
# CHECK-BE: efdmul 0, 3, 8                  # encoding: [0x10,0x03,0x42,0xe8]
# CHECK-LE: efdmul 0, 3, 8                  # encoding: [0xe8,0x42,0x03,0x10]
            efdmul %r0, %r3, %r8
# CHECK-BE: efdnabs 3, 23                   # encoding: [0x10,0x77,0x02,0xe5]
# CHECK-LE: efdnabs 3, 23                   # encoding: [0xe5,0x02,0x77,0x10]
            efdnabs %r3, %r23
# CHECK-BE: efdneg 3, 22                    # encoding: [0x10,0x76,0x02,0xe6]
# CHECK-LE: efdneg 3, 22                    # encoding: [0xe6,0x02,0x76,0x10]
            efdneg %r3, %r22
# CHECK-BE: efdsub 3, 4, 6                  # encoding: [0x10,0x64,0x32,0xe1]
# CHECK-LE: efdsub 3, 4, 6                  # encoding: [0xe1,0x32,0x64,0x10]
            efdsub %r3, %r4, %r6
# CHECK-BE: efdtsteq 3, 4, 5                # encoding: [0x11,0x84,0x2a,0xfe]
# CHECK-LE: efdtsteq 3, 4, 5                # encoding: [0xfe,0x2a,0x84,0x11]
            efdtsteq %cr3, %r4, %r5
# CHECK-BE: efdtstgt 3, 3, 6                # encoding: [0x11,0x83,0x32,0xfc]
# CHECK-LE: efdtstgt 3, 3, 6                # encoding: [0xfc,0x32,0x83,0x11]
            efdtstgt %cr3, %r3, %r6
# CHECK-BE: efdtstlt 4, 0, 3                # encoding: [0x12,0x00,0x1a,0xfd]
# CHECK-LE: efdtstlt 4, 0, 3                # encoding: [0xfd,0x1a,0x00,0x12]
            efdtstlt %cr4, %r0, %r3
# CHECK-BE: efsabs 3, 4                     # encoding: [0x10,0x64,0x02,0xc4]
# CHECK-LE: efsabs 3, 4                     # encoding: [0xc4,0x02,0x64,0x10]
            efsabs %r3, %r4
# CHECK-BE: efsadd 3, 4, 5                  # encoding: [0x10,0x64,0x2a,0xc0]
# CHECK-LE: efsadd 3, 4, 5                  # encoding: [0xc0,0x2a,0x64,0x10]
            efsadd %r3, %r4, %r5
# CHECK-BE: efscfsf 5, 6                    # encoding: [0x10,0xa0,0x32,0xd3]
# CHECK-LE: efscfsf 5, 6                    # encoding: [0xd3,0x32,0xa0,0x10]
            efscfsf %r5, %r6
# CHECK-BE: efscfsi 5, 6                    # encoding: [0x10,0xa0,0x32,0xd1]
# CHECK-LE: efscfsi 5, 6                    # encoding: [0xd1,0x32,0xa0,0x10]
            efscfsi %r5, %r6
# CHECK-BE: efscfuf 5, 8                    # encoding: [0x10,0xa0,0x42,0xd2]
# CHECK-LE: efscfuf 5, 8                    # encoding: [0xd2,0x42,0xa0,0x10]
            efscfuf %r5, %r8
# CHECK-BE: efscfui 6, 9                    # encoding: [0x10,0xc0,0x4a,0xd0]
# CHECK-LE: efscfui 6, 9                    # encoding: [0xd0,0x4a,0xc0,0x10]
            efscfui %r6, %r9
# CHECK-BE: efscmpeq 3, 3, 8                # encoding: [0x11,0x83,0x42,0xce]
# CHECK-LE: efscmpeq 3, 3, 8                # encoding: [0xce,0x42,0x83,0x11]
            efscmpeq %cr3, %r3, %r8
# CHECK-BE: efscmpgt 4, 7, 3                # encoding: [0x12,0x07,0x1a,0xcc]
# CHECK-LE: efscmpgt 4, 7, 3                # encoding: [0xcc,0x1a,0x07,0x12]
            efscmpgt %cr4, %r7, %r3
# CHECK-BE: efscmplt 2, 3, 4                # encoding: [0x11,0x03,0x22,0xcd]
# CHECK-LE: efscmplt 2, 3, 4                # encoding: [0xcd,0x22,0x03,0x11]
            efscmplt %cr2, %r3, %r4
# CHECK-BE: efsctsf 5, 3                    # encoding: [0x10,0xa0,0x1a,0xd7]
# CHECK-LE: efsctsf 5, 3                    # encoding: [0xd7,0x1a,0xa0,0x10]
            efsctsf %r5, %r3
# CHECK-BE: efsctsi 6, 4                    # encoding: [0x10,0xc0,0x22,0xd5]
# CHECK-LE: efsctsi 6, 4                    # encoding: [0xd5,0x22,0xc0,0x10]
            efsctsi %r6, %r4
# CHECK-BE: efsctsiz 3, 4                   # encoding: [0x10,0x60,0x22,0xda]
# CHECK-LE: efsctsiz 3, 4                   # encoding: [0xda,0x22,0x60,0x10]
            efsctsiz %r3, %r4
# CHECK-BE: efsctuf 5, 8                    # encoding: [0x10,0xa0,0x42,0xd6]
# CHECK-LE: efsctuf 5, 8                    # encoding: [0xd6,0x42,0xa0,0x10]
            efsctuf %r5, %r8
# CHECK-BE: efsctui 9, 10                   # encoding: [0x11,0x20,0x52,0xd4]
# CHECK-LE: efsctui 9, 10                   # encoding: [0xd4,0x52,0x20,0x11]
            efsctui %r9, %r10
# CHECK-BE: efsctuiz 5, 17                  # encoding: [0x10,0xa0,0x8a,0xd8]
# CHECK-LE: efsctuiz 5, 17                  # encoding: [0xd8,0x8a,0xa0,0x10]
            efsctuiz %r5, %r17
# CHECK-BE: efsdiv 3, 4, 5                  # encoding: [0x10,0x64,0x2a,0xc9]
# CHECK-LE: efsdiv 3, 4, 5                  # encoding: [0xc9,0x2a,0x64,0x10]
            efsdiv %r3, %r4, %r5
# CHECK-BE: efsmul 0, 3, 8                  # encoding: [0x10,0x03,0x42,0xc8]
# CHECK-LE: efsmul 0, 3, 8                  # encoding: [0xc8,0x42,0x03,0x10]
            efsmul %r0, %r3, %r8
# CHECK-BE: efsnabs 3, 23                   # encoding: [0x10,0x77,0x02,0xc5]
# CHECK-LE: efsnabs 3, 23                   # encoding: [0xc5,0x02,0x77,0x10]
            efsnabs %r3, %r23
# CHECK-BE: efsneg 3, 22                    # encoding: [0x10,0x76,0x02,0xc6]
# CHECK-LE: efsneg 3, 22                    # encoding: [0xc6,0x02,0x76,0x10]
            efsneg %r3, %r22
# CHECK-BE: efssub 3, 4, 6                  # encoding: [0x10,0x64,0x32,0xc1]
# CHECK-LE: efssub 3, 4, 6                  # encoding: [0xc1,0x32,0x64,0x10]
            efssub %r3, %r4, %r6
# CHECK-BE: efststeq 3, 4, 5                # encoding: [0x11,0x84,0x2a,0xde]
# CHECK-LE: efststeq 3, 4, 5                # encoding: [0xde,0x2a,0x84,0x11]
            efststeq %cr3, %r4, %r5
# CHECK-BE: efststgt 3, 3, 6                # encoding: [0x11,0x83,0x32,0xdc]
# CHECK-LE: efststgt 3, 3, 6                # encoding: [0xdc,0x32,0x83,0x11]
            efststgt %cr3, %r3, %r6
# CHECK-BE: efststlt 4, 0, 3                # encoding: [0x12,0x00,0x1a,0xdd]
# CHECK-LE: efststlt 4, 0, 3                # encoding: [0xdd,0x1a,0x00,0x12]
            efststlt %cr4, %r0, %r3
