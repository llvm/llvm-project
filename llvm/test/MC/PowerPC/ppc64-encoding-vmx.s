
# RUN: llvm-mc -triple powerpc64-unknown-unknown --show-encoding %s | FileCheck -check-prefix=CHECK-BE %s 
# RUN: llvm-mc -triple powerpc64le-unknown-unknown --show-encoding %s | FileCheck -check-prefix=CHECK-LE %s

# Vector facility

# Vector storage access instructions

# CHECK-BE: lvebx 2, 3, 4                   # encoding: [0x7c,0x43,0x20,0x0e]
# CHECK-LE: lvebx 2, 3, 4                   # encoding: [0x0e,0x20,0x43,0x7c]
            lvebx 2, 3, 4
# CHECK-BE: lvehx 2, 3, 4                   # encoding: [0x7c,0x43,0x20,0x4e]
# CHECK-LE: lvehx 2, 3, 4                   # encoding: [0x4e,0x20,0x43,0x7c]
            lvehx 2, 3, 4
# CHECK-BE: lvewx 2, 3, 4                   # encoding: [0x7c,0x43,0x20,0x8e]
# CHECK-LE: lvewx 2, 3, 4                   # encoding: [0x8e,0x20,0x43,0x7c]
            lvewx 2, 3, 4
# CHECK-BE: lvx 2, 3, 4                     # encoding: [0x7c,0x43,0x20,0xce]
# CHECK-LE: lvx 2, 3, 4                     # encoding: [0xce,0x20,0x43,0x7c]
            lvx 2, 3, 4
# CHECK-BE: lvxl 2, 3, 4                    # encoding: [0x7c,0x43,0x22,0xce]
# CHECK-LE: lvxl 2, 3, 4                    # encoding: [0xce,0x22,0x43,0x7c]
            lvxl 2, 3, 4
# CHECK-BE: stvebx 2, 3, 4                  # encoding: [0x7c,0x43,0x21,0x0e]
# CHECK-LE: stvebx 2, 3, 4                  # encoding: [0x0e,0x21,0x43,0x7c]
            stvebx 2, 3, 4
# CHECK-BE: stvehx 2, 3, 4                  # encoding: [0x7c,0x43,0x21,0x4e]
# CHECK-LE: stvehx 2, 3, 4                  # encoding: [0x4e,0x21,0x43,0x7c]
            stvehx 2, 3, 4
# CHECK-BE: stvewx 2, 3, 4                  # encoding: [0x7c,0x43,0x21,0x8e]
# CHECK-LE: stvewx 2, 3, 4                  # encoding: [0x8e,0x21,0x43,0x7c]
            stvewx 2, 3, 4
# CHECK-BE: stvx 2, 3, 4                    # encoding: [0x7c,0x43,0x21,0xce]
# CHECK-LE: stvx 2, 3, 4                    # encoding: [0xce,0x21,0x43,0x7c]
            stvx 2, 3, 4
# CHECK-BE: stvxl 2, 3, 4                   # encoding: [0x7c,0x43,0x23,0xce]
# CHECK-LE: stvxl 2, 3, 4                   # encoding: [0xce,0x23,0x43,0x7c]
            stvxl 2, 3, 4
# CHECK-BE: lvsl 2, 3, 4                    # encoding: [0x7c,0x43,0x20,0x0c]
# CHECK-LE: lvsl 2, 3, 4                    # encoding: [0x0c,0x20,0x43,0x7c]
            lvsl 2, 3, 4
# CHECK-BE: lvsr 2, 3, 4                    # encoding: [0x7c,0x43,0x20,0x4c]
# CHECK-LE: lvsr 2, 3, 4                    # encoding: [0x4c,0x20,0x43,0x7c]
            lvsr 2, 3, 4

# Vector permute and formatting instructions

# CHECK-BE: vpkpx 2, 3, 4                   # encoding: [0x10,0x43,0x23,0x0e]
# CHECK-LE: vpkpx 2, 3, 4                   # encoding: [0x0e,0x23,0x43,0x10]
            vpkpx 2, 3, 4
# CHECK-BE: vpkshss 2, 3, 4                 # encoding: [0x10,0x43,0x21,0x8e]
# CHECK-LE: vpkshss 2, 3, 4                 # encoding: [0x8e,0x21,0x43,0x10]
            vpkshss 2, 3, 4
# CHECK-BE: vpkshus 2, 3, 4                 # encoding: [0x10,0x43,0x21,0x0e]
# CHECK-LE: vpkshus 2, 3, 4                 # encoding: [0x0e,0x21,0x43,0x10]
            vpkshus 2, 3, 4
# CHECK-BE: vpkswss 2, 3, 4                 # encoding: [0x10,0x43,0x21,0xce]
# CHECK-LE: vpkswss 2, 3, 4                 # encoding: [0xce,0x21,0x43,0x10]
            vpkswss 2, 3, 4
# CHECK-BE: vpkswus 2, 3, 4                 # encoding: [0x10,0x43,0x21,0x4e]
# CHECK-LE: vpkswus 2, 3, 4                 # encoding: [0x4e,0x21,0x43,0x10]
            vpkswus 2, 3, 4
# CHECK-BE: vpkuhum 2, 3, 4                 # encoding: [0x10,0x43,0x20,0x0e]
# CHECK-LE: vpkuhum 2, 3, 4                 # encoding: [0x0e,0x20,0x43,0x10]
            vpkuhum 2, 3, 4
# CHECK-BE: vpkuhus 2, 3, 4                 # encoding: [0x10,0x43,0x20,0x8e]
# CHECK-LE: vpkuhus 2, 3, 4                 # encoding: [0x8e,0x20,0x43,0x10]
            vpkuhus 2, 3, 4
# CHECK-BE: vpkuwum 2, 3, 4                 # encoding: [0x10,0x43,0x20,0x4e]
# CHECK-LE: vpkuwum 2, 3, 4                 # encoding: [0x4e,0x20,0x43,0x10]
            vpkuwum 2, 3, 4
# CHECK-BE: vpkuwus 2, 3, 4                 # encoding: [0x10,0x43,0x20,0xce]
# CHECK-LE: vpkuwus 2, 3, 4                 # encoding: [0xce,0x20,0x43,0x10]
            vpkuwus 2, 3, 4

# CHECK-BE: vupkhpx 2, 3                    # encoding: [0x10,0x40,0x1b,0x4e]
# CHECK-LE: vupkhpx 2, 3                    # encoding: [0x4e,0x1b,0x40,0x10]
            vupkhpx 2, 3
# CHECK-BE: vupkhsb 2, 3                    # encoding: [0x10,0x40,0x1a,0x0e]
# CHECK-LE: vupkhsb 2, 3                    # encoding: [0x0e,0x1a,0x40,0x10]
            vupkhsb 2, 3
# CHECK-BE: vupkhsh 2, 3                    # encoding: [0x10,0x40,0x1a,0x4e]
# CHECK-LE: vupkhsh 2, 3                    # encoding: [0x4e,0x1a,0x40,0x10]
            vupkhsh 2, 3
# CHECK-BE: vupklpx 2, 3                    # encoding: [0x10,0x40,0x1b,0xce]
# CHECK-LE: vupklpx 2, 3                    # encoding: [0xce,0x1b,0x40,0x10]
            vupklpx 2, 3
# CHECK-BE: vupklsb 2, 3                    # encoding: [0x10,0x40,0x1a,0x8e]
# CHECK-LE: vupklsb 2, 3                    # encoding: [0x8e,0x1a,0x40,0x10]
            vupklsb 2, 3
# CHECK-BE: vupklsh 2, 3                    # encoding: [0x10,0x40,0x1a,0xce]
# CHECK-LE: vupklsh 2, 3                    # encoding: [0xce,0x1a,0x40,0x10]
            vupklsh 2, 3

# CHECK-BE: vmrghb 2, 3, 4                  # encoding: [0x10,0x43,0x20,0x0c]
# CHECK-LE: vmrghb 2, 3, 4                  # encoding: [0x0c,0x20,0x43,0x10]
            vmrghb 2, 3, 4
# CHECK-BE: vmrghh 2, 3, 4                  # encoding: [0x10,0x43,0x20,0x4c]
# CHECK-LE: vmrghh 2, 3, 4                  # encoding: [0x4c,0x20,0x43,0x10]
            vmrghh 2, 3, 4
# CHECK-BE: vmrghw 2, 3, 4                  # encoding: [0x10,0x43,0x20,0x8c]
# CHECK-LE: vmrghw 2, 3, 4                  # encoding: [0x8c,0x20,0x43,0x10]
            vmrghw 2, 3, 4
# CHECK-BE: vmrglb 2, 3, 4                  # encoding: [0x10,0x43,0x21,0x0c]
# CHECK-LE: vmrglb 2, 3, 4                  # encoding: [0x0c,0x21,0x43,0x10]
            vmrglb 2, 3, 4
# CHECK-BE: vmrglh 2, 3, 4                  # encoding: [0x10,0x43,0x21,0x4c]
# CHECK-LE: vmrglh 2, 3, 4                  # encoding: [0x4c,0x21,0x43,0x10]
            vmrglh 2, 3, 4
# CHECK-BE: vmrglw 2, 3, 4                  # encoding: [0x10,0x43,0x21,0x8c]
# CHECK-LE: vmrglw 2, 3, 4                  # encoding: [0x8c,0x21,0x43,0x10]
            vmrglw 2, 3, 4
# CHECK-BE: vmrgew 2, 3, 4                  # encoding: [0x10,0x43,0x27,0x8c]
# CHECK-LE: vmrgew 2, 3, 4                  # encoding: [0x8c,0x27,0x43,0x10]
            vmrgew 2, 3, 4
# CHECK-BE: vmrgow 2, 3, 4                  # encoding: [0x10,0x43,0x26,0x8c]
# CHECK-LE: vmrgow 2, 3, 4                  # encoding: [0x8c,0x26,0x43,0x10]
            vmrgow 2, 3, 4
	
# CHECK-BE: vspltb 2, 3, 1                  # encoding: [0x10,0x41,0x1a,0x0c]
# CHECK-LE: vspltb 2, 3, 1                  # encoding: [0x0c,0x1a,0x41,0x10]
            vspltb 2, 3, 1
# CHECK-BE: vsplth 2, 3, 1                  # encoding: [0x10,0x41,0x1a,0x4c]
# CHECK-LE: vsplth 2, 3, 1                  # encoding: [0x4c,0x1a,0x41,0x10]
            vsplth 2, 3, 1
# CHECK-BE: vspltw 2, 3, 1                  # encoding: [0x10,0x41,0x1a,0x8c]
# CHECK-LE: vspltw 2, 3, 1                  # encoding: [0x8c,0x1a,0x41,0x10]
            vspltw 2, 3, 1
# CHECK-BE: vspltisb 2, 3                   # encoding: [0x10,0x43,0x03,0x0c]
# CHECK-LE: vspltisb 2, 3                   # encoding: [0x0c,0x03,0x43,0x10]
            vspltisb 2, 3
# CHECK-BE: vspltish 2, 3                   # encoding: [0x10,0x43,0x03,0x4c]
# CHECK-LE: vspltish 2, 3                   # encoding: [0x4c,0x03,0x43,0x10]
            vspltish 2, 3
# CHECK-BE: vspltisw 2, 3                   # encoding: [0x10,0x43,0x03,0x8c]
# CHECK-LE: vspltisw 2, 3                   # encoding: [0x8c,0x03,0x43,0x10]
            vspltisw 2, 3

# CHECK-BE: vperm 2, 3, 4, 5                # encoding: [0x10,0x43,0x21,0x6b]
# CHECK-LE: vperm 2, 3, 4, 5                # encoding: [0x6b,0x21,0x43,0x10]
            vperm 2, 3, 4, 5

# CHECK-BE: vpermxor 2, 3, 4, 5             # encoding: [0x10,0x43,0x21,0x6d]
# CHECK-LE: vpermxor 2, 3, 4, 5             # encoding: [0x6d,0x21,0x43,0x10]
            vpermxor 2, 3, 4, 5

# CHECK-BE: vsbox 2, 5                      # encoding: [0x10,0x45,0x05,0xc8]
# CHECK-LE: vsbox 2, 5                      # encoding: [0xc8,0x05,0x45,0x10]
            vsbox 2, 5

# CHECK-BE: vcipher 2, 5, 17                # encoding: [0x10,0x45,0x8d,0x08]
# CHECK-LE: vcipher 2, 5, 17                # encoding: [0x08,0x8d,0x45,0x10]
            vcipher 2, 5, 17

# CHECK-BE: vcipherlast 2, 5, 17            # encoding: [0x10,0x45,0x8d,0x09]
# CHECK-LE: vcipherlast 2, 5, 17            # encoding: [0x09,0x8d,0x45,0x10]
            vcipherlast 2, 5, 17

# CHECK-BE: vncipher 2, 5, 17               # encoding: [0x10,0x45,0x8d,0x48]
# CHECK-LE: vncipher 2, 5, 17               # encoding: [0x48,0x8d,0x45,0x10]
            vncipher 2, 5, 17

# CHECK-BE: vncipherlast 2, 5, 17           # encoding: [0x10,0x45,0x8d,0x49]
# CHECK-LE: vncipherlast 2, 5, 17           # encoding: [0x49,0x8d,0x45,0x10]
            vncipherlast 2, 5, 17

# CHECK-BE: vpmsumb 2, 5, 17                # encoding: [0x10,0x45,0x8c,0x08]
# CHECK-LE: vpmsumb 2, 5, 17                # encoding: [0x08,0x8c,0x45,0x10]
            vpmsumb 2, 5, 17

# CHECK-BE: vpmsumh 2, 5, 17                # encoding: [0x10,0x45,0x8c,0x48]
# CHECK-LE: vpmsumh 2, 5, 17                # encoding: [0x48,0x8c,0x45,0x10]
            vpmsumh 2, 5, 17

# CHECK-BE: vpmsumw 2, 5, 17                # encoding: [0x10,0x45,0x8c,0x88]
# CHECK-LE: vpmsumw 2, 5, 17                # encoding: [0x88,0x8c,0x45,0x10]
            vpmsumw 2, 5, 17

# CHECK-BE: vpmsumd 2, 5, 17                # encoding: [0x10,0x45,0x8c,0xc8]
# CHECK-LE: vpmsumd 2, 5, 17                # encoding: [0xc8,0x8c,0x45,0x10]
            vpmsumd 2, 5, 17

# CHECK-BE: vshasigmaw 2, 3, 0, 11          # encoding: [0x10,0x43,0x5e,0x82]
# CHECK-LE: vshasigmaw 2, 3, 0, 11          # encoding: [0x82,0x5e,0x43,0x10]
            vshasigmaw 2, 3, 0, 11

# CHECK-BE: vshasigmad 2, 3, 1, 15          # encoding: [0x10,0x43,0xfe,0xc2]
# CHECK-LE: vshasigmad 2, 3, 1, 15          # encoding: [0xc2,0xfe,0x43,0x10]
            vshasigmad 2, 3, 1, 15

# CHECK-BE: vsel 2, 3, 4, 5                 # encoding: [0x10,0x43,0x21,0x6a]
# CHECK-LE: vsel 2, 3, 4, 5                 # encoding: [0x6a,0x21,0x43,0x10]
            vsel 2, 3, 4, 5

# CHECK-BE: vsl 2, 3, 4                     # encoding: [0x10,0x43,0x21,0xc4]
# CHECK-LE: vsl 2, 3, 4                     # encoding: [0xc4,0x21,0x43,0x10]
            vsl 2, 3, 4
# CHECK-BE: vsldoi 2, 3, 4, 5               # encoding: [0x10,0x43,0x21,0x6c]
# CHECK-LE: vsldoi 2, 3, 4, 5               # encoding: [0x6c,0x21,0x43,0x10]
            vsldoi 2, 3, 4, 5
# CHECK-BE: vslo 2, 3, 4                    # encoding: [0x10,0x43,0x24,0x0c]
# CHECK-LE: vslo 2, 3, 4                    # encoding: [0x0c,0x24,0x43,0x10]
            vslo 2, 3, 4
# CHECK-BE: vsr 2, 3, 4                     # encoding: [0x10,0x43,0x22,0xc4]
# CHECK-LE: vsr 2, 3, 4                     # encoding: [0xc4,0x22,0x43,0x10]
            vsr 2, 3, 4
# CHECK-BE: vsro 2, 3, 4                    # encoding: [0x10,0x43,0x24,0x4c]
# CHECK-LE: vsro 2, 3, 4                    # encoding: [0x4c,0x24,0x43,0x10]
            vsro 2, 3, 4

# Vector integer arithmetic instructions

# CHECK-BE: vaddcuw 2, 3, 4                 # encoding: [0x10,0x43,0x21,0x80]
# CHECK-LE: vaddcuw 2, 3, 4                 # encoding: [0x80,0x21,0x43,0x10]
            vaddcuw 2, 3, 4
# CHECK-BE: vaddsbs 2, 3, 4                 # encoding: [0x10,0x43,0x23,0x00]
# CHECK-LE: vaddsbs 2, 3, 4                 # encoding: [0x00,0x23,0x43,0x10]
            vaddsbs 2, 3, 4
# CHECK-BE: vaddshs 2, 3, 4                 # encoding: [0x10,0x43,0x23,0x40]
# CHECK-LE: vaddshs 2, 3, 4                 # encoding: [0x40,0x23,0x43,0x10]
            vaddshs 2, 3, 4
# CHECK-BE: vaddsws 2, 3, 4                 # encoding: [0x10,0x43,0x23,0x80]
# CHECK-LE: vaddsws 2, 3, 4                 # encoding: [0x80,0x23,0x43,0x10]
            vaddsws 2, 3, 4
# CHECK-BE: vaddubm 2, 3, 4                 # encoding: [0x10,0x43,0x20,0x00]
# CHECK-LE: vaddubm 2, 3, 4                 # encoding: [0x00,0x20,0x43,0x10]
            vaddubm 2, 3, 4
# CHECK-BE: vadduhm 2, 3, 4                 # encoding: [0x10,0x43,0x20,0x40]
# CHECK-LE: vadduhm 2, 3, 4                 # encoding: [0x40,0x20,0x43,0x10]
            vadduhm 2, 3, 4
# CHECK-BE: vadduwm 2, 3, 4                 # encoding: [0x10,0x43,0x20,0x80]
# CHECK-LE: vadduwm 2, 3, 4                 # encoding: [0x80,0x20,0x43,0x10]
            vadduwm 2, 3, 4
# CHECK-BE: vaddudm 2, 3, 4                 # encoding: [0x10,0x43,0x20,0xc0]
# CHECK-LE: vaddudm 2, 3, 4                 # encoding: [0xc0,0x20,0x43,0x10]
            vaddudm 2, 3, 4
# CHECK-BE: vaddubs 2, 3, 4                 # encoding: [0x10,0x43,0x22,0x00]
# CHECK-LE: vaddubs 2, 3, 4                 # encoding: [0x00,0x22,0x43,0x10]
            vaddubs 2, 3, 4
# CHECK-BE: vadduhs 2, 3, 4                 # encoding: [0x10,0x43,0x22,0x40]
# CHECK-LE: vadduhs 2, 3, 4                 # encoding: [0x40,0x22,0x43,0x10]
            vadduhs 2, 3, 4
# CHECK-BE: vadduws 2, 3, 4                 # encoding: [0x10,0x43,0x22,0x80]
# CHECK-LE: vadduws 2, 3, 4                 # encoding: [0x80,0x22,0x43,0x10]
            vadduws 2, 3, 4
# CHECK-BE: vadduqm 2, 3, 4                 # encoding: [0x10,0x43,0x21,0x00]
# CHECK-LE: vadduqm 2, 3, 4                 # encoding: [0x00,0x21,0x43,0x10]
            vadduqm 2, 3, 4
# CHECK-BE: vaddeuqm 2, 3, 4, 5             # encoding: [0x10,0x43,0x21,0x7c]
# CHECK-LE: vaddeuqm 2, 3, 4, 5             # encoding:	[0x7c,0x21,0x43,0x10]
            vaddeuqm 2, 3, 4, 5
# CHECK-BE: vaddcuq 2, 3, 4                 # encoding: [0x10,0x43,0x21,0x40]
# CHECK-LE: vaddcuq 2, 3, 4                 # encoding: [0x40,0x21,0x43,0x10]
            vaddcuq 2, 3, 4
# CHECK-BE: vaddecuq 2, 3, 4, 5             # encoding: [0x10,0x43,0x21,0x7d]
# CHECK-LE: vaddecuq 2, 3, 4, 5             # encoding: [0x7d,0x21,0x43,0x10]
            vaddecuq 2, 3, 4, 5   
	
# CHECK-BE: vsubcuw 2, 3, 4                 # encoding: [0x10,0x43,0x25,0x80]
# CHECK-LE: vsubcuw 2, 3, 4                 # encoding: [0x80,0x25,0x43,0x10]
            vsubcuw 2, 3, 4
# CHECK-BE: vsubsbs 2, 3, 4                 # encoding: [0x10,0x43,0x27,0x00]
# CHECK-LE: vsubsbs 2, 3, 4                 # encoding: [0x00,0x27,0x43,0x10]
            vsubsbs 2, 3, 4
# CHECK-BE: vsubshs 2, 3, 4                 # encoding: [0x10,0x43,0x27,0x40]
# CHECK-LE: vsubshs 2, 3, 4                 # encoding: [0x40,0x27,0x43,0x10]
            vsubshs 2, 3, 4
# CHECK-BE: vsubsws 2, 3, 4                 # encoding: [0x10,0x43,0x27,0x80]
# CHECK-LE: vsubsws 2, 3, 4                 # encoding: [0x80,0x27,0x43,0x10]
            vsubsws 2, 3, 4
# CHECK-BE: vsububm 2, 3, 4                 # encoding: [0x10,0x43,0x24,0x00]
# CHECK-LE: vsububm 2, 3, 4                 # encoding: [0x00,0x24,0x43,0x10]
            vsububm 2, 3, 4
# CHECK-BE: vsubuhm 2, 3, 4                 # encoding: [0x10,0x43,0x24,0x40]
# CHECK-LE: vsubuhm 2, 3, 4                 # encoding: [0x40,0x24,0x43,0x10]
            vsubuhm 2, 3, 4
# CHECK-BE: vsubuwm 2, 3, 4                 # encoding: [0x10,0x43,0x24,0x80]
# CHECK-LE: vsubuwm 2, 3, 4                 # encoding: [0x80,0x24,0x43,0x10]
            vsubuwm 2, 3, 4
# CHECK-BE: vsubudm 2, 3, 4                 # encoding: [0x10,0x43,0x24,0xc0]
# CHECK-LE: vsubudm 2, 3, 4                 # encoding: [0xc0,0x24,0x43,0x10]
            vsubudm 2, 3, 4
# CHECK-BE: vsububs 2, 3, 4                 # encoding: [0x10,0x43,0x26,0x00]
# CHECK-LE: vsububs 2, 3, 4                 # encoding: [0x00,0x26,0x43,0x10]
            vsububs 2, 3, 4
# CHECK-BE: vsubuhs 2, 3, 4                 # encoding: [0x10,0x43,0x26,0x40]
# CHECK-LE: vsubuhs 2, 3, 4                 # encoding: [0x40,0x26,0x43,0x10]
            vsubuhs 2, 3, 4
# CHECK-BE: vsubuws 2, 3, 4                 # encoding: [0x10,0x43,0x26,0x80]
# CHECK-LE: vsubuws 2, 3, 4                 # encoding: [0x80,0x26,0x43,0x10]
            vsubuws 2, 3, 4
# CHECK-BE: vsubuqm 2, 3, 4                 # encoding: [0x10,0x43,0x25,0x00]
# CHECK-LE: vsubuqm 2, 3, 4                 # encoding: [0x00,0x25,0x43,0x10]
            vsubuqm 2, 3, 4
# CHECK-BE: vsubeuqm 2, 3, 4, 5             # encoding: [0x10,0x43,0x21,0x7e]
# CHECK-LE: vsubeuqm 2, 3, 4, 5             # encoding: [0x7e,0x21,0x43,0x10]
            vsubeuqm 2, 3, 4, 5
# CHECK-BE: vsubcuq 2, 3, 4                 # encoding: [0x10,0x43,0x25,0x40] 
# CHECK-LE: vsubcuq 2, 3, 4                 # encoding: [0x40,0x25,0x43,0x10]
            vsubcuq 2, 3, 4
# CHECK-BE: vsubecuq 2, 3, 4, 5             # encoding: [0x10,0x43,0x21,0x7f]
# CHECK-LE: vsubecuq 2, 3, 4, 5             # encoding: [0x7f,0x21,0x43,0x10]
            vsubecuq 2, 3, 4, 5	
	
# CHECK-BE: vmulesb 2, 3, 4                 # encoding: [0x10,0x43,0x23,0x08]
# CHECK-LE: vmulesb 2, 3, 4                 # encoding: [0x08,0x23,0x43,0x10]
            vmulesb 2, 3, 4
# CHECK-BE: vmulesh 2, 3, 4                 # encoding: [0x10,0x43,0x23,0x48]
# CHECK-LE: vmulesh 2, 3, 4                 # encoding: [0x48,0x23,0x43,0x10]
            vmulesh 2, 3, 4
# CHECK-BE: vmulesw 2, 3, 4                 # encoding: [0x10,0x43,0x23,0x88]
# CHECK-LE: vmulesw 2, 3, 4                 # encoding: [0x88,0x23,0x43,0x10]
            vmulesw 2, 3, 4
# CHECK-BE: vmuleub 2, 3, 4                 # encoding: [0x10,0x43,0x22,0x08]
# CHECK-LE: vmuleub 2, 3, 4                 # encoding: [0x08,0x22,0x43,0x10]
            vmuleub 2, 3, 4
# CHECK-BE: vmuleuh 2, 3, 4                 # encoding: [0x10,0x43,0x22,0x48]
# CHECK-LE: vmuleuh 2, 3, 4                 # encoding: [0x48,0x22,0x43,0x10]
            vmuleuh 2, 3, 4
# CHECK-BE: vmuleuw 2, 3, 4                 # encoding: [0x10,0x43,0x22,0x88]
# CHECK-LE: vmuleuw 2, 3, 4                 # encoding: [0x88,0x22,0x43,0x10]
            vmuleuw 2, 3, 4
# CHECK-BE: vmulosb 2, 3, 4                 # encoding: [0x10,0x43,0x21,0x08]
# CHECK-LE: vmulosb 2, 3, 4                 # encoding: [0x08,0x21,0x43,0x10]
            vmulosb 2, 3, 4
# CHECK-BE: vmulosh 2, 3, 4                 # encoding: [0x10,0x43,0x21,0x48]
# CHECK-LE: vmulosh 2, 3, 4                 # encoding: [0x48,0x21,0x43,0x10]
            vmulosh 2, 3, 4
# CHECK-BE: vmulosw 2, 3, 4                 # encoding: [0x10,0x43,0x21,0x88]
# CHECK-LE: vmulosw 2, 3, 4                 # encoding: [0x88,0x21,0x43,0x10]
            vmulosw 2, 3, 4
# CHECK-BE: vmuloub 2, 3, 4                 # encoding: [0x10,0x43,0x20,0x08]
# CHECK-LE: vmuloub 2, 3, 4                 # encoding: [0x08,0x20,0x43,0x10]
            vmuloub 2, 3, 4
# CHECK-BE: vmulouh 2, 3, 4                 # encoding: [0x10,0x43,0x20,0x48]
# CHECK-LE: vmulouh 2, 3, 4                 # encoding: [0x48,0x20,0x43,0x10]
            vmulouh 2, 3, 4
# CHECK-BE: vmulouw 2, 3, 4                 # encoding: [0x10,0x43,0x20,0x88]
# CHECK-LE: vmulouw 2, 3, 4                 # encoding: [0x88,0x20,0x43,0x10]
            vmulouw 2, 3, 4
# CHECK-BE: vmuluwm 2, 3, 4                 # encoding: [0x10,0x43,0x20,0x89]
# CHECK-LE: vmuluwm 2, 3, 4                 # encoding: [0x89,0x20,0x43,0x10]
            vmuluwm 2, 3, 4

# CHECK-BE: vmhaddshs 2, 3, 4, 5            # encoding: [0x10,0x43,0x21,0x60]
# CHECK-LE: vmhaddshs 2, 3, 4, 5            # encoding: [0x60,0x21,0x43,0x10]
            vmhaddshs 2, 3, 4, 5
# CHECK-BE: vmhraddshs 2, 3, 4, 5           # encoding: [0x10,0x43,0x21,0x61]
# CHECK-LE: vmhraddshs 2, 3, 4, 5           # encoding: [0x61,0x21,0x43,0x10]
            vmhraddshs 2, 3, 4, 5
# CHECK-BE: vmladduhm 2, 3, 4, 5            # encoding: [0x10,0x43,0x21,0x62]
# CHECK-LE: vmladduhm 2, 3, 4, 5            # encoding: [0x62,0x21,0x43,0x10]
            vmladduhm 2, 3, 4, 5
# CHECK-BE: vmsumubm 2, 3, 4, 5             # encoding: [0x10,0x43,0x21,0x64]
# CHECK-LE: vmsumubm 2, 3, 4, 5             # encoding: [0x64,0x21,0x43,0x10]
            vmsumubm 2, 3, 4, 5
# CHECK-BE: vmsummbm 2, 3, 4, 5             # encoding: [0x10,0x43,0x21,0x65]
# CHECK-LE: vmsummbm 2, 3, 4, 5             # encoding: [0x65,0x21,0x43,0x10]
            vmsummbm 2, 3, 4, 5
# CHECK-BE: vmsumshm 2, 3, 4, 5             # encoding: [0x10,0x43,0x21,0x68]
# CHECK-LE: vmsumshm 2, 3, 4, 5             # encoding: [0x68,0x21,0x43,0x10]
            vmsumshm 2, 3, 4, 5
# CHECK-BE: vmsumshs 2, 3, 4, 5             # encoding: [0x10,0x43,0x21,0x69]
# CHECK-LE: vmsumshs 2, 3, 4, 5             # encoding: [0x69,0x21,0x43,0x10]
            vmsumshs 2, 3, 4, 5
# CHECK-BE: vmsumuhm 2, 3, 4, 5             # encoding: [0x10,0x43,0x21,0x66]
# CHECK-LE: vmsumuhm 2, 3, 4, 5             # encoding: [0x66,0x21,0x43,0x10]
            vmsumuhm 2, 3, 4, 5
# CHECK-BE: vmsumuhs 2, 3, 4, 5             # encoding: [0x10,0x43,0x21,0x67]
# CHECK-LE: vmsumuhs 2, 3, 4, 5             # encoding: [0x67,0x21,0x43,0x10]
            vmsumuhs 2, 3, 4, 5

# CHECK-BE: vsumsws 2, 3, 4                 # encoding: [0x10,0x43,0x27,0x88]
# CHECK-LE: vsumsws 2, 3, 4                 # encoding: [0x88,0x27,0x43,0x10]
            vsumsws 2, 3, 4
# CHECK-BE: vsum2sws 2, 3, 4                # encoding: [0x10,0x43,0x26,0x88]
# CHECK-LE: vsum2sws 2, 3, 4                # encoding: [0x88,0x26,0x43,0x10]
            vsum2sws 2, 3, 4
# CHECK-BE: vsum4sbs 2, 3, 4                # encoding: [0x10,0x43,0x27,0x08]
# CHECK-LE: vsum4sbs 2, 3, 4                # encoding: [0x08,0x27,0x43,0x10]
            vsum4sbs 2, 3, 4
# CHECK-BE: vsum4shs 2, 3, 4                # encoding: [0x10,0x43,0x26,0x48]
# CHECK-LE: vsum4shs 2, 3, 4                # encoding: [0x48,0x26,0x43,0x10]
            vsum4shs 2, 3, 4
# CHECK-BE: vsum4ubs 2, 3, 4                # encoding: [0x10,0x43,0x26,0x08]
# CHECK-LE: vsum4ubs 2, 3, 4                # encoding: [0x08,0x26,0x43,0x10]
            vsum4ubs 2, 3, 4

# CHECK-BE: vavgsb 2, 3, 4                  # encoding: [0x10,0x43,0x25,0x02]
# CHECK-LE: vavgsb 2, 3, 4                  # encoding: [0x02,0x25,0x43,0x10]
            vavgsb 2, 3, 4
# CHECK-BE: vavgsh 2, 3, 4                  # encoding: [0x10,0x43,0x25,0x42]
# CHECK-LE: vavgsh 2, 3, 4                  # encoding: [0x42,0x25,0x43,0x10]
            vavgsh 2, 3, 4
# CHECK-BE: vavgsw 2, 3, 4                  # encoding: [0x10,0x43,0x25,0x82]
# CHECK-LE: vavgsw 2, 3, 4                  # encoding: [0x82,0x25,0x43,0x10]
            vavgsw 2, 3, 4
# CHECK-BE: vavgub 2, 3, 4                  # encoding: [0x10,0x43,0x24,0x02]
# CHECK-LE: vavgub 2, 3, 4                  # encoding: [0x02,0x24,0x43,0x10]
            vavgub 2, 3, 4
# CHECK-BE: vavguh 2, 3, 4                  # encoding: [0x10,0x43,0x24,0x42]
# CHECK-LE: vavguh 2, 3, 4                  # encoding: [0x42,0x24,0x43,0x10]
            vavguh 2, 3, 4
# CHECK-BE: vavguw 2, 3, 4                  # encoding: [0x10,0x43,0x24,0x82]
# CHECK-LE: vavguw 2, 3, 4                  # encoding: [0x82,0x24,0x43,0x10]
            vavguw 2, 3, 4

# CHECK-BE: vmaxsb 2, 3, 4                  # encoding: [0x10,0x43,0x21,0x02]
# CHECK-LE: vmaxsb 2, 3, 4                  # encoding: [0x02,0x21,0x43,0x10]
            vmaxsb 2, 3, 4
# CHECK-BE: vmaxsh 2, 3, 4                  # encoding: [0x10,0x43,0x21,0x42]
# CHECK-LE: vmaxsh 2, 3, 4                  # encoding: [0x42,0x21,0x43,0x10]
            vmaxsh 2, 3, 4
# CHECK-BE: vmaxsw 2, 3, 4                  # encoding: [0x10,0x43,0x21,0x82]
# CHECK-LE: vmaxsw 2, 3, 4                  # encoding: [0x82,0x21,0x43,0x10]
            vmaxsw 2, 3, 4
# CHECK-BE: vmaxsd 2, 3, 4                  # encoding: [0x10,0x43,0x21,0xc2]
# CHECK-LE: vmaxsd 2, 3, 4                    # encoding: [0xc2,0x21,0x43,0x10]
            vmaxsd 2, 3, 4
# CHECK-BE: vmaxub 2, 3, 4                  # encoding: [0x10,0x43,0x20,0x02]
# CHECK-LE: vmaxub 2, 3, 4                  # encoding: [0x02,0x20,0x43,0x10]
            vmaxub 2, 3, 4
# CHECK-BE: vmaxuh 2, 3, 4                  # encoding: [0x10,0x43,0x20,0x42]
# CHECK-LE: vmaxuh 2, 3, 4                  # encoding: [0x42,0x20,0x43,0x10]
            vmaxuh 2, 3, 4
# CHECK-BE: vmaxuw 2, 3, 4                  # encoding: [0x10,0x43,0x20,0x82]
# CHECK-LE: vmaxuw 2, 3, 4                  # encoding: [0x82,0x20,0x43,0x10]
            vmaxuw 2, 3, 4
# CHECK-BE: vmaxud 2, 3, 4                  # encoding: [0x10,0x43,0x20,0xc2]
# CHECK-LE: vmaxud 2, 3, 4                  # encoding: [0xc2,0x20,0x43,0x10]
            vmaxud 2, 3, 4
        
# CHECK-BE: vminsb 2, 3, 4                  # encoding: [0x10,0x43,0x23,0x02]
# CHECK-LE: vminsb 2, 3, 4                  # encoding: [0x02,0x23,0x43,0x10]
            vminsb 2, 3, 4
# CHECK-BE: vminsh 2, 3, 4                  # encoding: [0x10,0x43,0x23,0x42]
# CHECK-LE: vminsh 2, 3, 4                  # encoding: [0x42,0x23,0x43,0x10]
            vminsh 2, 3, 4
# CHECK-BE: vminsw 2, 3, 4                  # encoding: [0x10,0x43,0x23,0x82]
# CHECK-LE: vminsw 2, 3, 4                  # encoding: [0x82,0x23,0x43,0x10]
            vminsw 2, 3, 4
# CHECK-BE: vminsd 2, 3, 4                  # encoding: [0x10,0x43,0x23,0xc2]
# CHECK-LE: vminsd 2, 3, 4                  # encoding: [0xc2,0x23,0x43,0x10]
            vminsd 2, 3, 4
# CHECK-BE: vminub 2, 3, 4                  # encoding: [0x10,0x43,0x22,0x02]
# CHECK-LE: vminub 2, 3, 4                  # encoding: [0x02,0x22,0x43,0x10]
            vminub 2, 3, 4
# CHECK-BE: vminuh 2, 3, 4                  # encoding: [0x10,0x43,0x22,0x42]
# CHECK-LE: vminuh 2, 3, 4                  # encoding: [0x42,0x22,0x43,0x10]
            vminuh 2, 3, 4
# CHECK-BE: vminuw 2, 3, 4                  # encoding: [0x10,0x43,0x22,0x82]
# CHECK-LE: vminuw 2, 3, 4                  # encoding: [0x82,0x22,0x43,0x10]
            vminuw 2, 3, 4
# CHECK-BE: vminud 2, 3, 4                  # encoding: [0x10,0x43,0x22,0xc2]
# CHECK-LE: vminud 2, 3, 4                  # encoding: [0xc2,0x22,0x43,0x10]
            vminud 2, 3, 4

# Vector integer compare instructions

# CHECK-BE: vcmpequb 2, 3, 4                # encoding: [0x10,0x43,0x20,0x06]
# CHECK-LE: vcmpequb 2, 3, 4                # encoding: [0x06,0x20,0x43,0x10]
            vcmpequb 2, 3, 4
# CHECK-BE: vcmpequb. 2, 3, 4               # encoding: [0x10,0x43,0x24,0x06]
# CHECK-LE: vcmpequb. 2, 3, 4               # encoding: [0x06,0x24,0x43,0x10]
            vcmpequb. 2, 3, 4
# CHECK-BE: vcmpequh 2, 3, 4                # encoding: [0x10,0x43,0x20,0x46]
# CHECK-LE: vcmpequh 2, 3, 4                # encoding: [0x46,0x20,0x43,0x10]
            vcmpequh 2, 3, 4
# CHECK-BE: vcmpequh. 2, 3, 4               # encoding: [0x10,0x43,0x24,0x46]
# CHECK-LE: vcmpequh. 2, 3, 4               # encoding: [0x46,0x24,0x43,0x10]
            vcmpequh. 2, 3, 4
# CHECK-BE: vcmpequw 2, 3, 4                # encoding: [0x10,0x43,0x20,0x86]
# CHECK-LE: vcmpequw 2, 3, 4                # encoding: [0x86,0x20,0x43,0x10]
            vcmpequw 2, 3, 4
# CHECK-BE: vcmpequw. 2, 3, 4               # encoding: [0x10,0x43,0x24,0x86]
# CHECK-LE: vcmpequw. 2, 3, 4               # encoding: [0x86,0x24,0x43,0x10]
            vcmpequw. 2, 3, 4
# CHECK-BE: vcmpequd 2, 3, 4                # encoding: [0x10,0x43,0x20,0xc7]
# CHECK-LE: vcmpequd 2, 3, 4                # encoding: [0xc7,0x20,0x43,0x10]
            vcmpequd 2, 3, 4
# CHECK-BE: vcmpequd. 2, 3, 4               # encoding: [0x10,0x43,0x24,0xc7]
# CHECK-LE: vcmpequd. 2, 3, 4               # encoding: [0xc7,0x24,0x43,0x10]
            vcmpequd. 2, 3, 4
# CHECK-BE: vcmpgtsb 2, 3, 4                # encoding: [0x10,0x43,0x23,0x06]
# CHECK-LE: vcmpgtsb 2, 3, 4                # encoding: [0x06,0x23,0x43,0x10]
            vcmpgtsb 2, 3, 4
# CHECK-BE: vcmpgtsb. 2, 3, 4               # encoding: [0x10,0x43,0x27,0x06]
# CHECK-LE: vcmpgtsb. 2, 3, 4               # encoding: [0x06,0x27,0x43,0x10]
            vcmpgtsb. 2, 3, 4
# CHECK-BE: vcmpgtsh 2, 3, 4                # encoding: [0x10,0x43,0x23,0x46]
# CHECK-LE: vcmpgtsh 2, 3, 4                # encoding: [0x46,0x23,0x43,0x10]
            vcmpgtsh 2, 3, 4
# CHECK-BE: vcmpgtsh. 2, 3, 4               # encoding: [0x10,0x43,0x27,0x46]
# CHECK-LE: vcmpgtsh. 2, 3, 4               # encoding: [0x46,0x27,0x43,0x10]
            vcmpgtsh. 2, 3, 4
# CHECK-BE: vcmpgtsw 2, 3, 4                # encoding: [0x10,0x43,0x23,0x86]
# CHECK-LE: vcmpgtsw 2, 3, 4                # encoding: [0x86,0x23,0x43,0x10]
            vcmpgtsw 2, 3, 4
# CHECK-BE: vcmpgtsw. 2, 3, 4               # encoding: [0x10,0x43,0x27,0x86]
# CHECK-LE: vcmpgtsw. 2, 3, 4               # encoding: [0x86,0x27,0x43,0x10]
            vcmpgtsw. 2, 3, 4
# CHECK-BE: vcmpgtsd 2, 3, 4                # encoding: [0x10,0x43,0x23,0xc7]
# CHECK-LE: vcmpgtsd 2, 3, 4                # encoding: [0xc7,0x23,0x43,0x10]
            vcmpgtsd 2, 3, 4
# CHECK-BE: vcmpgtsd. 2, 3, 4               # encoding: [0x10,0x43,0x27,0xc7]
# CHECK-LE: vcmpgtsd. 2, 3, 4               # encoding: [0xc7,0x27,0x43,0x10]
            vcmpgtsd. 2, 3, 4
# CHECK-BE: vcmpgtub 2, 3, 4                # encoding: [0x10,0x43,0x22,0x06]
# CHECK-LE: vcmpgtub 2, 3, 4                # encoding: [0x06,0x22,0x43,0x10]
            vcmpgtub 2, 3, 4
# CHECK-BE: vcmpgtub. 2, 3, 4               # encoding: [0x10,0x43,0x26,0x06]
# CHECK-LE: vcmpgtub. 2, 3, 4               # encoding: [0x06,0x26,0x43,0x10]
            vcmpgtub. 2, 3, 4
# CHECK-BE: vcmpgtuh 2, 3, 4                # encoding: [0x10,0x43,0x22,0x46]
# CHECK-LE: vcmpgtuh 2, 3, 4                # encoding: [0x46,0x22,0x43,0x10]
            vcmpgtuh 2, 3, 4
# CHECK-BE: vcmpgtuh. 2, 3, 4               # encoding: [0x10,0x43,0x26,0x46]
# CHECK-LE: vcmpgtuh. 2, 3, 4               # encoding: [0x46,0x26,0x43,0x10]
            vcmpgtuh. 2, 3, 4
# CHECK-BE: vcmpgtuw 2, 3, 4                # encoding: [0x10,0x43,0x22,0x86]
# CHECK-LE: vcmpgtuw 2, 3, 4                # encoding: [0x86,0x22,0x43,0x10]
            vcmpgtuw 2, 3, 4
# CHECK-BE: vcmpgtuw. 2, 3, 4               # encoding: [0x10,0x43,0x26,0x86]
# CHECK-LE: vcmpgtuw. 2, 3, 4               # encoding: [0x86,0x26,0x43,0x10]
            vcmpgtuw. 2, 3, 4
# CHECK-BE: vcmpgtud 2, 3, 4                # encoding: [0x10,0x43,0x22,0xc7]
# CHECK-LE: vcmpgtud 2, 3, 4                # encoding: [0xc7,0x22,0x43,0x10]
            vcmpgtud 2, 3, 4
# CHECK-BE: vcmpgtud. 2, 3, 4               # encoding: [0x10,0x43,0x26,0xc7]
# CHECK-LE: vcmpgtud. 2, 3, 4               # encoding: [0xc7,0x26,0x43,0x10]
            vcmpgtud. 2, 3, 4
        
# Vector integer logical instructions

# CHECK-BE: vand 2, 3, 4                    # encoding: [0x10,0x43,0x24,0x04]
# CHECK-LE: vand 2, 3, 4                    # encoding: [0x04,0x24,0x43,0x10]
            vand 2, 3, 4
# CHECK-BE: vandc 2, 3, 4                   # encoding: [0x10,0x43,0x24,0x44]
# CHECK-LE: vandc 2, 3, 4                   # encoding: [0x44,0x24,0x43,0x10]
            vandc 2, 3, 4
# CHECK-BE: veqv 2, 3, 4                    # encoding: [0x10,0x43,0x26,0x84]
# CHECK-LE: veqv 2, 3, 4                    # encoding: [0x84,0x26,0x43,0x10]
            veqv 2, 3, 4
# CHECK-BE: vnand 2, 3, 4                   # encoding: [0x10,0x43,0x25,0x84]
# CHECK-LE: vnand 2, 3, 4                   # encoding: [0x84,0x25,0x43,0x10]
            vnand 2, 3, 4
# CHECK-BE: vorc 2, 3, 4                    # encoding: [0x10,0x43,0x25,0x44]
# CHECK-LE: vorc 2, 3, 4                    # encoding: [0x44,0x25,0x43,0x10]
            vorc 2, 3, 4
# CHECK-BE: vnor 2, 3, 4                    # encoding: [0x10,0x43,0x25,0x04]
# CHECK-LE: vnor 2, 3, 4                    # encoding: [0x04,0x25,0x43,0x10]
            vnor 2, 3, 4
# CHECK-BE: vor 2, 3, 4                     # encoding: [0x10,0x43,0x24,0x84]
# CHECK-LE: vor 2, 3, 4                     # encoding: [0x84,0x24,0x43,0x10]
            vor 2, 3, 4
# CHECK-BE: vxor 2, 3, 4                    # encoding: [0x10,0x43,0x24,0xc4]
# CHECK-LE: vxor 2, 3, 4                    # encoding: [0xc4,0x24,0x43,0x10]
            vxor 2, 3, 4

# Vector integer rotate and shift instructions

# CHECK-BE: vrlb 2, 3, 4                    # encoding: [0x10,0x43,0x20,0x04]
# CHECK-LE: vrlb 2, 3, 4                    # encoding: [0x04,0x20,0x43,0x10]
            vrlb 2, 3, 4
# CHECK-BE: vrlh 2, 3, 4                    # encoding: [0x10,0x43,0x20,0x44]
# CHECK-LE: vrlh 2, 3, 4                    # encoding: [0x44,0x20,0x43,0x10]
            vrlh 2, 3, 4
# CHECK-BE: vrlw 2, 3, 4                    # encoding: [0x10,0x43,0x20,0x84]
# CHECK-LE: vrlw 2, 3, 4                    # encoding: [0x84,0x20,0x43,0x10]
            vrlw 2, 3, 4
# CHECK-BE: vrld 2, 3, 4                    # encoding: [0x10,0x43,0x20,0xc4]
# CHECK-LE: vrld 2, 3, 4                    # encoding: [0xc4,0x20,0x43,0x10]
            vrld 2, 3, 4
# CHECK-BE: vslb 2, 3, 4                    # encoding: [0x10,0x43,0x21,0x04]
# CHECK-LE: vslb 2, 3, 4                    # encoding: [0x04,0x21,0x43,0x10]
            vslb 2, 3, 4
# CHECK-BE: vslh 2, 3, 4                    # encoding: [0x10,0x43,0x21,0x44]
# CHECK-LE: vslh 2, 3, 4                    # encoding: [0x44,0x21,0x43,0x10]
            vslh 2, 3, 4
# CHECK-BE: vslw 2, 3, 4                    # encoding: [0x10,0x43,0x21,0x84]
# CHECK-LE: vslw 2, 3, 4                    # encoding: [0x84,0x21,0x43,0x10]
            vslw 2, 3, 4
# CHECK-BE: vsld 2, 3, 4                    # encoding: [0x10,0x43,0x25,0xc4]
# CHECK-LE: vsld 2, 3, 4                    # encoding: [0xc4,0x25,0x43,0x10]
            vsld 2, 3, 4
# CHECK-BE: vsrb 2, 3, 4                    # encoding: [0x10,0x43,0x22,0x04]
# CHECK-LE: vsrb 2, 3, 4                    # encoding: [0x04,0x22,0x43,0x10]
            vsrb 2, 3, 4
# CHECK-BE: vsrh 2, 3, 4                    # encoding: [0x10,0x43,0x22,0x44]
# CHECK-LE: vsrh 2, 3, 4                    # encoding: [0x44,0x22,0x43,0x10]
            vsrh 2, 3, 4
# CHECK-BE: vsrw 2, 3, 4                    # encoding: [0x10,0x43,0x22,0x84]
# CHECK-LE: vsrw 2, 3, 4                    # encoding: [0x84,0x22,0x43,0x10]
            vsrw 2, 3, 4
# CHECK-BE: vsrd 2, 3, 4                    # encoding: [0x10,0x43,0x26,0xc4]
# CHECK-LE: vsrd 2, 3, 4                    # encoding: [0xc4,0x26,0x43,0x10]
            vsrd 2, 3, 4
# CHECK-BE: vsrab 2, 3, 4                   # encoding: [0x10,0x43,0x23,0x04]
# CHECK-LE: vsrab 2, 3, 4                   # encoding: [0x04,0x23,0x43,0x10]
            vsrab 2, 3, 4
# CHECK-BE: vsrah 2, 3, 4                   # encoding: [0x10,0x43,0x23,0x44]
# CHECK-LE: vsrah 2, 3, 4                   # encoding: [0x44,0x23,0x43,0x10]
            vsrah 2, 3, 4
# CHECK-BE: vsraw 2, 3, 4                   # encoding: [0x10,0x43,0x23,0x84]
# CHECK-LE: vsraw 2, 3, 4                   # encoding: [0x84,0x23,0x43,0x10]
            vsraw 2, 3, 4
# CHECK-BE: vsrad 2, 3, 4                   # encoding: [0x10,0x43,0x23,0xc4]
# CHECK-LE: vsrad 2, 3, 4                   # encoding: [0xc4,0x23,0x43,0x10]
            vsrad 2, 3, 4

# Vector floating-point instructions

# CHECK-BE: vaddfp 2, 3, 4                  # encoding: [0x10,0x43,0x20,0x0a]
# CHECK-LE: vaddfp 2, 3, 4                  # encoding: [0x0a,0x20,0x43,0x10]
            vaddfp 2, 3, 4
# CHECK-BE: vsubfp 2, 3, 4                  # encoding: [0x10,0x43,0x20,0x4a]
# CHECK-LE: vsubfp 2, 3, 4                  # encoding: [0x4a,0x20,0x43,0x10]
            vsubfp 2, 3, 4
# CHECK-BE: vmaddfp 2, 3, 4, 5              # encoding: [0x10,0x43,0x29,0x2e]
# CHECK-LE: vmaddfp 2, 3, 4, 5              # encoding: [0x2e,0x29,0x43,0x10]
            vmaddfp 2, 3, 4, 5
# CHECK-BE: vnmsubfp 2, 3, 4, 5             # encoding: [0x10,0x43,0x29,0x2f]
# CHECK-LE: vnmsubfp 2, 3, 4, 5             # encoding: [0x2f,0x29,0x43,0x10]
            vnmsubfp 2, 3, 4, 5

# CHECK-BE: vmaxfp 2, 3, 4                  # encoding: [0x10,0x43,0x24,0x0a]
# CHECK-LE: vmaxfp 2, 3, 4                  # encoding: [0x0a,0x24,0x43,0x10]
            vmaxfp 2, 3, 4
# CHECK-BE: vminfp 2, 3, 4                  # encoding: [0x10,0x43,0x24,0x4a]
# CHECK-LE: vminfp 2, 3, 4                  # encoding: [0x4a,0x24,0x43,0x10]
            vminfp 2, 3, 4

# CHECK-BE: vctsxs 2, 3, 4                  # encoding: [0x10,0x44,0x1b,0xca]
# CHECK-LE: vctsxs 2, 3, 4                  # encoding: [0xca,0x1b,0x44,0x10]
            vctsxs 2, 3, 4
# CHECK-BE: vctuxs 2, 3, 4                  # encoding: [0x10,0x44,0x1b,0x8a]
# CHECK-LE: vctuxs 2, 3, 4                  # encoding: [0x8a,0x1b,0x44,0x10]
            vctuxs 2, 3, 4
# CHECK-BE: vcfsx 2, 3, 4                   # encoding: [0x10,0x44,0x1b,0x4a]
# CHECK-LE: vcfsx 2, 3, 4                   # encoding: [0x4a,0x1b,0x44,0x10]
            vcfsx 2, 3, 4
# CHECK-BE: vcfux 2, 3, 4                   # encoding: [0x10,0x44,0x1b,0x0a]
# CHECK-LE: vcfux 2, 3, 4                   # encoding: [0x0a,0x1b,0x44,0x10]
            vcfux 2, 3, 4
# CHECK-BE: vrfim 2, 3                      # encoding: [0x10,0x40,0x1a,0xca]
# CHECK-LE: vrfim 2, 3                      # encoding: [0xca,0x1a,0x40,0x10]
            vrfim 2, 3
# CHECK-BE: vrfin 2, 3                      # encoding: [0x10,0x40,0x1a,0x0a]
# CHECK-LE: vrfin 2, 3                      # encoding: [0x0a,0x1a,0x40,0x10]
            vrfin 2, 3
# CHECK-BE: vrfip 2, 3                      # encoding: [0x10,0x40,0x1a,0x8a]
# CHECK-LE: vrfip 2, 3                      # encoding: [0x8a,0x1a,0x40,0x10]
            vrfip 2, 3
# CHECK-BE: vrfiz 2, 3                      # encoding: [0x10,0x40,0x1a,0x4a]
# CHECK-LE: vrfiz 2, 3                      # encoding: [0x4a,0x1a,0x40,0x10]
            vrfiz 2, 3

# CHECK-BE: vcmpbfp 2, 3, 4                 # encoding: [0x10,0x43,0x23,0xc6]
# CHECK-LE: vcmpbfp 2, 3, 4                 # encoding: [0xc6,0x23,0x43,0x10]
            vcmpbfp 2, 3, 4
# CHECK-BE: vcmpbfp. 2, 3, 4                # encoding: [0x10,0x43,0x27,0xc6]
# CHECK-LE: vcmpbfp. 2, 3, 4                # encoding: [0xc6,0x27,0x43,0x10]
            vcmpbfp. 2, 3, 4
# CHECK-BE: vcmpeqfp 2, 3, 4                # encoding: [0x10,0x43,0x20,0xc6]
# CHECK-LE: vcmpeqfp 2, 3, 4                # encoding: [0xc6,0x20,0x43,0x10]
            vcmpeqfp 2, 3, 4
# CHECK-BE: vcmpeqfp. 2, 3, 4               # encoding: [0x10,0x43,0x24,0xc6]
# CHECK-LE: vcmpeqfp. 2, 3, 4               # encoding: [0xc6,0x24,0x43,0x10]
            vcmpeqfp. 2, 3, 4
# CHECK-BE: vcmpgefp 2, 3, 4                # encoding: [0x10,0x43,0x21,0xc6]
# CHECK-LE: vcmpgefp 2, 3, 4                # encoding: [0xc6,0x21,0x43,0x10]
            vcmpgefp 2, 3, 4
# CHECK-BE: vcmpgefp. 2, 3, 4               # encoding: [0x10,0x43,0x25,0xc6]
# CHECK-LE: vcmpgefp. 2, 3, 4               # encoding: [0xc6,0x25,0x43,0x10]
            vcmpgefp. 2, 3, 4
# CHECK-BE: vcmpgtfp 2, 3, 4                # encoding: [0x10,0x43,0x22,0xc6]
# CHECK-LE: vcmpgtfp 2, 3, 4                # encoding: [0xc6,0x22,0x43,0x10]
            vcmpgtfp 2, 3, 4
# CHECK-BE: vcmpgtfp. 2, 3, 4               # encoding: [0x10,0x43,0x26,0xc6]
# CHECK-LE: vcmpgtfp. 2, 3, 4               # encoding: [0xc6,0x26,0x43,0x10]
            vcmpgtfp. 2, 3, 4

# CHECK-BE: vexptefp 2, 3                   # encoding: [0x10,0x40,0x19,0x8a]
# CHECK-LE: vexptefp 2, 3                   # encoding: [0x8a,0x19,0x40,0x10]
            vexptefp 2, 3
# CHECK-BE: vlogefp 2, 3                    # encoding: [0x10,0x40,0x19,0xca]
# CHECK-LE: vlogefp 2, 3                    # encoding: [0xca,0x19,0x40,0x10]
            vlogefp 2, 3
# CHECK-BE: vrefp 2, 3                      # encoding: [0x10,0x40,0x19,0x0a]
# CHECK-LE: vrefp 2, 3                      # encoding: [0x0a,0x19,0x40,0x10]
            vrefp 2, 3
# CHECK-BE: vrsqrtefp 2, 3                  # encoding: [0x10,0x40,0x19,0x4a]
# CHECK-LE: vrsqrtefp 2, 3                  # encoding: [0x4a,0x19,0x40,0x10]
            vrsqrtefp 2, 3
# CHECK-BE: vgbbd 2, 3                      # encoding: [0x10,0x40,0x1d,0x0c]
# CHECK-LE: vgbbd 2, 3                      # encoding: [0x0c,0x1d,0x40,0x10]
            vgbbd 2, 3
# CHECK-BE: vbpermq 2, 5, 17                # encoding: [0x10,0x45,0x8d,0x4c]
# CHECK-LE: vbpermq 2, 5, 17                # encoding: [0x4c,0x8d,0x45,0x10]
            vbpermq 2, 5, 17

# Vector count leading zero instructions
# CHECK-BE: vclzb 2, 3                      # encoding: [0x10,0x40,0x1f,0x02]
# CHECK-LE: vclzb 2, 3                      # encoding: [0x02,0x1f,0x40,0x10]
            vclzb 2, 3

# CHECK-BE: vclzh 2, 3                      # encoding: [0x10,0x40,0x1f,0x42]
# CHECK-LE: vclzh 2, 3                      # encoding: [0x42,0x1f,0x40,0x10]
            vclzh 2, 3

# CHECK-BE: vclzw 2, 3                      # encoding: [0x10,0x40,0x1f,0x82]
# CHECK-LE: vclzw 2, 3                      # encoding: [0x82,0x1f,0x40,0x10]
            vclzw 2, 3

# CHECK-BE: vclzd 2, 3                      # encoding: [0x10,0x40,0x1f,0xc2]
# CHECK-LE: vclzd 2, 3                      # encoding: [0xc2,0x1f,0x40,0x10]
            vclzd 2, 3                      

# Vector population count instructions
# CHECK-BE: vpopcntb 2, 3                   # encoding: [0x10,0x40,0x1f,0x03]
# CHECK-LE: vpopcntb 2, 3                   # encoding: [0x03,0x1f,0x40,0x10]
            vpopcntb 2, 3

# CHECK-BE: vpopcnth 2, 3                   # encoding: [0x10,0x40,0x1f,0x43]
# CHECK-LE: vpopcnth 2, 3                   # encoding: [0x43,0x1f,0x40,0x10]
            vpopcnth 2, 3

# CHECK-BE: vpopcntw 2, 3                   # encoding: [0x10,0x40,0x1f,0x83]
# CHECK-LE: vpopcntw 2, 3                   # encoding: [0x83,0x1f,0x40,0x10]
            vpopcntw 2, 3
        
# BCHECK-BE: vpopcntd 2, 3                   # encoding: [0x10,0x40,0x1f,0xC3]
# BCHECK-LE: vpopcntd 2, 3                   # encoding: [0xC3,0x1f,0x40,0x10]
#            vpopcntd 2, 3
        
# Vector status and control register instructions

# CHECK-BE: mtvscr 2                        # encoding: [0x10,0x00,0x16,0x44]
# CHECK-LE: mtvscr 2                        # encoding: [0x44,0x16,0x00,0x10]
            mtvscr 2
# CHECK-BE: mfvscr 2                        # encoding: [0x10,0x40,0x06,0x04]
# CHECK-LE: mfvscr 2                        # encoding: [0x04,0x06,0x40,0x10]
            mfvscr 2

# Power9 instructions

# Vector Compare Not Equal (Zero)
# CHECK-BE: vcmpneb 2, 3, 4                 # encoding: [0x10,0x43,0x20,0x07]
# CHECK-LE: vcmpneb 2, 3, 4                 # encoding: [0x07,0x20,0x43,0x10]
            vcmpneb 2, 3, 4
# CHECK-BE: vcmpneb. 2, 3, 4                # encoding: [0x10,0x43,0x24,0x07]
# CHECK-LE: vcmpneb. 2, 3, 4                # encoding: [0x07,0x24,0x43,0x10]
            vcmpneb. 2, 3, 4
# CHECK-BE: vcmpnezb 2, 3, 4                # encoding: [0x10,0x43,0x21,0x07]
# CHECK-LE: vcmpnezb 2, 3, 4                # encoding: [0x07,0x21,0x43,0x10]
            vcmpnezb 2, 3, 4
# CHECK-BE: vcmpnezb. 2, 3, 4               # encoding: [0x10,0x43,0x25,0x07]
# CHECK-LE: vcmpnezb. 2, 3, 4               # encoding: [0x07,0x25,0x43,0x10]
            vcmpnezb. 2, 3, 4
# CHECK-BE: vcmpneh 2, 3, 4                 # encoding: [0x10,0x43,0x20,0x47]
# CHECK-LE: vcmpneh 2, 3, 4                 # encoding: [0x47,0x20,0x43,0x10]
            vcmpneh 2, 3, 4
# CHECK-BE: vcmpneh. 2, 3, 4                # encoding: [0x10,0x43,0x24,0x47]
# CHECK-LE: vcmpneh. 2, 3, 4                # encoding: [0x47,0x24,0x43,0x10]
            vcmpneh. 2, 3, 4
# CHECK-BE: vcmpnezh 2, 3, 4                # encoding: [0x10,0x43,0x21,0x47]
# CHECK-LE: vcmpnezh 2, 3, 4                # encoding: [0x47,0x21,0x43,0x10]
            vcmpnezh 2, 3, 4
# CHECK-BE: vcmpnezh. 2, 3, 4               # encoding: [0x10,0x43,0x25,0x47]
# CHECK-LE: vcmpnezh. 2, 3, 4               # encoding: [0x47,0x25,0x43,0x10]
            vcmpnezh. 2, 3, 4
# CHECK-BE: vcmpnew 2, 3, 4                 # encoding: [0x10,0x43,0x20,0x87]
# CHECK-LE: vcmpnew 2, 3, 4                 # encoding: [0x87,0x20,0x43,0x10]
            vcmpnew 2, 3, 4
# CHECK-BE: vcmpnew. 2, 3, 4                # encoding: [0x10,0x43,0x24,0x87]
# CHECK-LE: vcmpnew. 2, 3, 4                # encoding: [0x87,0x24,0x43,0x10]
            vcmpnew. 2, 3, 4
# CHECK-BE: vcmpnezw 2, 3, 4                # encoding: [0x10,0x43,0x21,0x87]
# CHECK-LE: vcmpnezw 2, 3, 4                # encoding: [0x87,0x21,0x43,0x10]
            vcmpnezw 2, 3, 4
# CHECK-BE: vcmpnezw. 2, 3, 4               # encoding: [0x10,0x43,0x25,0x87]
# CHECK-LE: vcmpnezw. 2, 3, 4               # encoding: [0x87,0x25,0x43,0x10]
            vcmpnezw. 2, 3, 4

# Vector Extract Unsigned
# CHECK-BE: vextractub 2, 3, 15             # encoding: [0x10,0x4f,0x1a,0x0d]
# CHECK-LE: vextractub 2, 3, 15             # encoding: [0x0d,0x1a,0x4f,0x10]
            vextractub 2, 3, 15
# CHECK-BE: vextractuh 2, 3, 14             # encoding: [0x10,0x4e,0x1a,0x4d]
# CHECK-LE: vextractuh 2, 3, 14             # encoding: [0x4d,0x1a,0x4e,0x10]
            vextractuh 2, 3, 14
# CHECK-BE: vextractuw 2, 3, 12             # encoding: [0x10,0x4c,0x1a,0x8d]
# CHECK-LE: vextractuw 2, 3, 12             # encoding: [0x8d,0x1a,0x4c,0x10]
            vextractuw 2, 3, 12
# CHECK-BE: vextractd 2, 3, 8               # encoding: [0x10,0x48,0x1a,0xcd]
# CHECK-LE: vextractd 2, 3, 8               # encoding: [0xcd,0x1a,0x48,0x10]
            vextractd 2, 3, 8

# Vector Extract Unsigned Left/Right-Indexed
# CHECK-BE: vextublx 2, 3, 4                # encoding: [0x10,0x43,0x26,0x0d]
# CHECK-LE: vextublx 2, 3, 4                # encoding: [0x0d,0x26,0x43,0x10]
            vextublx 2, 3, 4
# CHECK-BE: vextubrx 2, 3, 4                # encoding: [0x10,0x43,0x27,0x0d]
# CHECK-LE: vextubrx 2, 3, 4                # encoding: [0x0d,0x27,0x43,0x10]
            vextubrx 2, 3, 4
# CHECK-BE: vextuhlx 2, 3, 4                # encoding: [0x10,0x43,0x26,0x4d]
# CHECK-LE: vextuhlx 2, 3, 4                # encoding: [0x4d,0x26,0x43,0x10]
            vextuhlx 2, 3, 4
# CHECK-BE: vextuhrx 2, 3, 4                # encoding: [0x10,0x43,0x27,0x4d]
# CHECK-LE: vextuhrx 2, 3, 4                # encoding: [0x4d,0x27,0x43,0x10]
            vextuhrx 2, 3, 4
# CHECK-BE: vextuwlx 2, 3, 4                # encoding: [0x10,0x43,0x26,0x8d]
# CHECK-LE: vextuwlx 2, 3, 4                # encoding: [0x8d,0x26,0x43,0x10]
            vextuwlx 2, 3, 4
# CHECK-BE: vextuwrx 2, 3, 4                # encoding: [0x10,0x43,0x27,0x8d]
# CHECK-LE: vextuwrx 2, 3, 4                # encoding: [0x8d,0x27,0x43,0x10]
            vextuwrx 2, 3, 4

# Vector Insert Element
# CHECK-BE: vinsertb 2, 3, 15               # encoding: [0x10,0x4f,0x1b,0x0d]
# CHECK-LE: vinsertb 2, 3, 15               # encoding: [0x0d,0x1b,0x4f,0x10]
            vinsertb 2, 3, 15
# CHECK-BE: vinserth 2, 3, 14               # encoding: [0x10,0x4e,0x1b,0x4d]
# CHECK-LE: vinserth 2, 3, 14               # encoding: [0x4d,0x1b,0x4e,0x10]
            vinserth 2, 3, 14
# CHECK-BE: vinsertw 2, 3, 12               # encoding: [0x10,0x4c,0x1b,0x8d]
# CHECK-LE: vinsertw 2, 3, 12               # encoding: [0x8d,0x1b,0x4c,0x10]
            vinsertw 2, 3, 12
# CHECK-BE: vinsertd 2, 3, 8                # encoding: [0x10,0x48,0x1b,0xcd]
# CHECK-LE: vinsertd 2, 3, 8                # encoding: [0xcd,0x1b,0x48,0x10]
            vinsertd 2, 3, 8

