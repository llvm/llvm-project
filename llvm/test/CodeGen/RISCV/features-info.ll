; RUN: llc -mtriple=riscv32 -mattr=help 2>&1 | FileCheck %s
; RUN: llc -mtriple=riscv64 -mattr=help 2>&1 | FileCheck %s

; CHECK: Available features for this target:
; CHECK:   32bit                            - Implements RV32.
; CHECK:   64bit                            - Implements RV64.
; CHECK:   a                                - 'A' (Atomic Instructions).
; CHECK:   auipc-addi-fusion                - Enable AUIPC+ADDI macrofusion.
; CHECK:   b                                - 'B' (the collection of the Zba, Zbb, Zbs extensions).
; CHECK:   c                                - 'C' (Compressed Instructions).
; CHECK:   conditional-cmv-fusion           - Enable branch+c.mv fusion.
; CHECK:   d                                - 'D' (Double-Precision Floating-Point).
; CHECK:   disable-latency-sched-heuristic  - Disable latency scheduling heuristic.
; CHECK:   dlen-factor-2                    - Vector unit DLEN(data path width) is half of VLEN.
; CHECK:   e                                - 'E' (Embedded Instruction Set with 16 GPRs).
; CHECK:   experimental                     - Experimental intrinsics.
; CHECK:   experimental-rvm23u32            - RISC-V experimental-rvm23u32 profile.
; CHECK:   experimental-sdext               - 'Sdext' (External debugger).
; CHECK:   experimental-sdtrig              - 'Sdtrig' (Debugger triggers).
; CHECK:   experimental-smctr               - 'Smctr' (Control Transfer Records Machine Level).
; CHECK:   experimental-ssctr               - 'Ssctr' (Control Transfer Records Supervisor Level).
; CHECK:   experimental-svukte              - 'Svukte' (Address-Independent Latency of User-Mode Faults to Supervisor Addresses).
; CHECK:   experimental-xqcia               - 'Xqcia' (Qualcomm uC Arithmetic Extension).
; CHECK:   experimental-xqcics              - 'Xqcics' (Qualcomm uC Conditional Select Extension).
; CHECK:   experimental-xqcicsr             - 'Xqcicsr' (Qualcomm uC CSR Extension).
; CHECK:   experimental-xqcilsm             - 'Xqcilsm' (Qualcomm uC Load Store Multiple Extension).
; CHECK:   experimental-xqcisls             - 'Xqcisls' (Qualcomm uC Scaled Load Store Extension).
; CHECK:   experimental-zalasr              - 'Zalasr' (Load-Acquire and Store-Release Instructions).
; CHECK:   experimental-zicfilp             - 'Zicfilp' (Landing pad).
; CHECK:   experimental-zicfiss             - 'Zicfiss' (Shadow stack).
; CHECK:   experimental-zvbc32e             - 'Zvbc32e' (Vector Carryless Multiplication with 32-bits elements).
; CHECK:   experimental-zvkgs               - 'Zvkgs' (Vector-Scalar GCM instructions for Cryptography).
; CHECK:   f                                - 'F' (Single-Precision Floating-Point).
; CHECK:   forced-atomics                   - Assume that lock-free native-width atomics are available.
; CHECK:   h                                - 'H' (Hypervisor).
; CHECK:   i                                - 'I' (Base Integer Instruction Set).
; CHECK:   ld-add-fusion                    - Enable LD+ADD macrofusion.
; CHECK:   lui-addi-fusion                  - Enable LUI+ADDI macro fusion.
; CHECK:   m                                - 'M' (Integer Multiplication and Division).
; CHECK:   mips-p8700                       - MIPS p8700 processor.
; CHECK:   no-default-unroll                - Disable default unroll preference..
; CHECK:   no-rvc-hints                     - Disable RVC Hint Instructions..
; CHECK:   no-sink-splat-operands           - Disable sink splat operands to enable .vx, .vf,.wx, and .wf instructions.
; CHECK:   no-trailing-seq-cst-fence        - Disable trailing fence for seq-cst store..
; CHECK:   optimized-nf2-segment-load-store - vlseg2eN.v and vsseg2eN.v areimplemented as a wide memory op and shuffle.
; CHECK:   optimized-nf3-segment-load-store - vlseg3eN.v and vsseg3eN.v areimplemented as a wide memory op and shuffle.
; CHECK:   optimized-nf4-segment-load-store - vlseg4eN.v and vsseg4eN.v areimplemented as a wide memory op and shuffle.
; CHECK:   optimized-nf5-segment-load-store - vlseg5eN.v and vsseg5eN.v areimplemented as a wide memory op and shuffle.
; CHECK:   optimized-nf6-segment-load-store - vlseg6eN.v and vsseg6eN.v areimplemented as a wide memory op and shuffle.
; CHECK:   optimized-nf7-segment-load-store - vlseg7eN.v and vsseg7eN.v areimplemented as a wide memory op and shuffle.
; CHECK:   optimized-nf8-segment-load-store - vlseg8eN.v and vsseg8eN.v areimplemented as a wide memory op and shuffle.
; CHECK:   optimized-zero-stride-load       - Optimized (perform fewer memory operations)zero-stride vector load.
; CHECK:   predictable-select-expensive     - Prefer likely predicted branches over selects.
; CHECK:   prefer-w-inst                    - Prefer instructions with W suffix.
; CHECK:   relax                            - Enable Linker relaxation..
; CHECK:   reserve-x1                       - Reserve X1.
; CHECK:   reserve-x10                      - Reserve X10.
; CHECK:   reserve-x11                      - Reserve X11.
; CHECK:   reserve-x12                      - Reserve X12.
; CHECK:   reserve-x13                      - Reserve X13.
; CHECK:   reserve-x14                      - Reserve X14.
; CHECK:   reserve-x15                      - Reserve X15.
; CHECK:   reserve-x16                      - Reserve X16.
; CHECK:   reserve-x17                      - Reserve X17.
; CHECK:   reserve-x18                      - Reserve X18.
; CHECK:   reserve-x19                      - Reserve X19.
; CHECK:   reserve-x2                       - Reserve X2.
; CHECK:   reserve-x20                      - Reserve X20.
; CHECK:   reserve-x21                      - Reserve X21.
; CHECK:   reserve-x22                      - Reserve X22.
; CHECK:   reserve-x23                      - Reserve X23.
; CHECK:   reserve-x24                      - Reserve X24.
; CHECK:   reserve-x25                      - Reserve X25.
; CHECK:   reserve-x26                      - Reserve X26.
; CHECK:   reserve-x27                      - Reserve X27.
; CHECK:   reserve-x28                      - Reserve X28.
; CHECK:   reserve-x29                      - Reserve X29.
; CHECK:   reserve-x3                       - Reserve X3.
; CHECK:   reserve-x30                      - Reserve X30.
; CHECK:   reserve-x31                      - Reserve X31.
; CHECK:   reserve-x4                       - Reserve X4.
; CHECK:   reserve-x5                       - Reserve X5.
; CHECK:   reserve-x6                       - Reserve X6.
; CHECK:   reserve-x7                       - Reserve X7.
; CHECK:   reserve-x8                       - Reserve X8.
; CHECK:   reserve-x9                       - Reserve X9.
; CHECK:   rva20s64                         - RISC-V rva20s64 profile.
; CHECK:   rva20u64                         - RISC-V rva20u64 profile.
; CHECK:   rva22s64                         - RISC-V rva22s64 profile.
; CHECK:   rva22u64                         - RISC-V rva22u64 profile.
; CHECK:   rva23s64                         - RISC-V rva23s64 profile.
; CHECK:   rva23u64                         - RISC-V rva23u64 profile.
; CHECK:   rvb23s64                         - RISC-V rvb23s64 profile.
; CHECK:   rvb23u64                         - RISC-V rvb23u64 profile.
; CHECK:   rvi20u32                         - RISC-V rvi20u32 profile.
; CHECK:   rvi20u64                         - RISC-V rvi20u64 profile.
; CHECK:   save-restore                     - Enable save/restore..
; CHECK:   sha                              - 'Sha' (Augmented Hypervisor).
; CHECK:   shcounterenw                     - 'Shcounterenw' (Support writeable hcounteren enable bit for any hpmcounter that is not read-only zero).
; CHECK:   shgatpa                          - 'Shgatpa' (SvNNx4 mode supported for all modes supported by satp, as well as Bare).
; CHECK:   shifted-zextw-fusion             - Enable SLLI+SRLI to be fused when computing (shifted) word zero extension.
; CHECK:   short-forward-branch-opt         - Enable short forward branch optimization.
; CHECK:   shtvala                          - 'Shtvala' (htval provides all needed values).
; CHECK:   shvsatpa                         - 'Shvsatpa' (vsatp supports all modes supported by satp).
; CHECK:   shvstvala                        - 'Shvstvala' (vstval provides all needed values).
; CHECK:   shvstvecd                        - 'Shvstvecd' (vstvec supports Direct mode).
; CHECK:   sifive7                          - SiFive 7-Series processors.
; CHECK:   smaia                            - 'Smaia' (Advanced Interrupt Architecture Machine Level).
; CHECK:   smcdeleg                         - 'Smcdeleg' (Counter Delegation Machine Level).
; CHECK:   smcsrind                         - 'Smcsrind' (Indirect CSR Access Machine Level).
; CHECK:   smdbltrp                         - 'Smdbltrp' (Double Trap Machine Level).
; CHECK:   smepmp                           - 'Smepmp' (Enhanced Physical Memory Protection).
; CHECK:   smmpm                            - 'Smmpm' (Machine-level Pointer Masking for M-mode).
; CHECK:   smnpm                            - 'Smnpm' (Machine-level Pointer Masking for next lower privilege mode).
; CHECK:   smrnmi                           - 'Smrnmi' (Resumable Non-Maskable Interrupts).
; CHECK:   smstateen                        - 'Smstateen' (Machine-mode view of the state-enable extension).
; CHECK:   ssaia                            - 'Ssaia' (Advanced Interrupt Architecture Supervisor Level).
; CHECK:   ssccfg                           - 'Ssccfg' (Counter Configuration Supervisor Level).
; CHECK:   ssccptr                          - 'Ssccptr' (Main memory supports page table reads).
; CHECK:   sscofpmf                         - 'Sscofpmf' (Count Overflow and Mode-Based Filtering).
; CHECK:   sscounterenw                     - 'Sscounterenw' (Support writeable scounteren enable bit for any hpmcounter that is not read-only zero).
; CHECK:   sscsrind                         - 'Sscsrind' (Indirect CSR Access Supervisor Level).
; CHECK:   ssdbltrp                         - 'Ssdbltrp' (Double Trap Supervisor Level).
; CHECK:   ssnpm                            - 'Ssnpm' (Supervisor-level Pointer Masking for next lower privilege mode).
; CHECK:   sspm                             - 'Sspm' (Indicates Supervisor-mode Pointer Masking).
; CHECK:   ssqosid                          - 'Ssqosid' (Quality-of-Service (QoS) Identifiers).
; CHECK:   ssstateen                        - 'Ssstateen' (Supervisor-mode view of the state-enable extension).
; CHECK:   ssstrict                         - 'Ssstrict' (No non-conforming extensions are present).
; CHECK:   sstc                             - 'Sstc' (Supervisor-mode timer interrupts).
; CHECK:   sstvala                          - 'Sstvala' (stval provides all needed values).
; CHECK:   sstvecd                          - 'Sstvecd' (stvec supports Direct mode).
; CHECK:   ssu64xl                          - 'Ssu64xl' (UXLEN=64 supported).
; CHECK:   supm                             - 'Supm' (Indicates User-mode Pointer Masking).
; CHECK:   svade                            - 'Svade' (Raise exceptions on improper A/D bits).
; CHECK:   svadu                            - 'Svadu' (Hardware A/D updates).
; CHECK:   svbare                           - 'Svbare' (satp mode Bare supported).
; CHECK:   svinval                          - 'Svinval' (Fine-Grained Address-Translation Cache Invalidation).
; CHECK:   svnapot                          - 'Svnapot' (NAPOT Translation Contiguity).
; CHECK:   svpbmt                           - 'Svpbmt' (Page-Based Memory Types).
; CHECK:   svvptc                           - 'Svvptc' (Obviating Memory-Management Instructions after Marking PTEs Valid).
; CHECK:   tagged-globals                   - Use an instruction sequence for taking the address of a global that allows a memory tag in the upper address bits.
; CHECK:   unaligned-scalar-mem             - Has reasonably performant unaligned scalar loads and stores.
; CHECK:   unaligned-vector-mem             - Has reasonably performant unaligned vector loads and stores.
; CHECK:   use-postra-scheduler             - Schedule again after register allocation.
; CHECK:   v                                - 'V' (Vector Extension for Application Processors).
; CHECK:   ventana-veyron                   - Ventana Veyron-Series processors.
; CHECK:   vxrm-pipeline-flush              - VXRM writes causes pipeline flush.
; CHECK:   xcvalu                           - 'XCValu' (CORE-V ALU Operations).
; CHECK:   xcvbi                            - 'XCVbi' (CORE-V Immediate Branching).
; CHECK:   xcvbitmanip                      - 'XCVbitmanip' (CORE-V Bit Manipulation).
; CHECK:   xcvelw                           - 'XCVelw' (CORE-V Event Load Word).
; CHECK:   xcvmac                           - 'XCVmac' (CORE-V Multiply-Accumulate).
; CHECK:   xcvmem                           - 'XCVmem' (CORE-V Post-incrementing Load & Store).
; CHECK:   xcvsimd                          - 'XCVsimd' (CORE-V SIMD ALU).
; CHECK:   xsfcease                         - 'XSfcease' (SiFive sf.cease Instruction).
; CHECK:   xsfvcp                           - 'XSfvcp' (SiFive Custom Vector Coprocessor Interface Instructions).
; CHECK:   xsfvfnrclipxfqf                  - 'XSfvfnrclipxfqf' (SiFive FP32-to-int8 Ranged Clip Instructions).
; CHECK:   xsfvfwmaccqqq                    - 'XSfvfwmaccqqq' (SiFive Matrix Multiply Accumulate Instruction and 4-by-4)).
; CHECK:   xsfvqmaccdod                     - 'XSfvqmaccdod' (SiFive Int8 Matrix Multiplication Instructions (2-by-8 and 8-by-2)).
; CHECK:   xsfvqmaccqoq                     - 'XSfvqmaccqoq' (SiFive Int8 Matrix Multiplication Instructions (4-by-8 and 8-by-4)).
; CHECK:   xsifivecdiscarddlone             - 'XSiFivecdiscarddlone' (SiFive sf.cdiscard.d.l1 Instruction).
; CHECK:   xsifivecflushdlone               - 'XSiFivecflushdlone' (SiFive sf.cflush.d.l1 Instruction).
; CHECK:   xtheadba                         - 'XTHeadBa' (T-Head address calculation instructions).
; CHECK:   xtheadbb                         - 'XTHeadBb' (T-Head basic bit-manipulation instructions).
; CHECK:   xtheadbs                         - 'XTHeadBs' (T-Head single-bit instructions).
; CHECK:   xtheadcmo                        - 'XTHeadCmo' (T-Head cache management instructions).
; CHECK:   xtheadcondmov                    - 'XTHeadCondMov' (T-Head conditional move instructions).
; CHECK:   xtheadfmemidx                    - 'XTHeadFMemIdx' (T-Head FP Indexed Memory Operations).
; CHECK:   xtheadmac                        - 'XTHeadMac' (T-Head Multiply-Accumulate Instructions).
; CHECK:   xtheadmemidx                     - 'XTHeadMemIdx' (T-Head Indexed Memory Operations).
; CHECK:   xtheadmempair                    - 'XTHeadMemPair' (T-Head two-GPR Memory Operations).
; CHECK:   xtheadsync                       - 'XTHeadSync' (T-Head multicore synchronization instructions).
; CHECK:   xtheadvdot                       - 'XTHeadVdot' (T-Head Vector Extensions for Dot).
; CHECK:   xventanacondops                  - 'XVentanaCondOps' (Ventana Conditional Ops).
; CHECK:   xwchc                            - 'Xwchc' (WCH/QingKe additional compressed opcodes).
; CHECK:   za128rs                          - 'Za128rs' (Reservation Set Size of at Most 128 Bytes).
; CHECK:   za64rs                           - 'Za64rs' (Reservation Set Size of at Most 64 Bytes).
; CHECK:   zaamo                            - 'Zaamo' (Atomic Memory Operations).
; CHECK:   zabha                            - 'Zabha' (Byte and Halfword Atomic Memory Operations).
; CHECK:   zacas                            - 'Zacas' (Atomic Compare-And-Swap Instructions).
; CHECK:   zalrsc                           - 'Zalrsc' (Load-Reserved/Store-Conditional).
; CHECK:   zama16b                          - 'Zama16b' (Atomic 16-byte misaligned loads, stores and AMOs).
; CHECK:   zawrs                            - 'Zawrs' (Wait on Reservation Set).
; CHECK:   zba                              - 'Zba' (Address Generation Instructions).
; CHECK:   zbb                              - 'Zbb' (Basic Bit-Manipulation).
; CHECK:   zbc                              - 'Zbc' (Carry-Less Multiplication).
; CHECK:   zbkb                             - 'Zbkb' (Bitmanip instructions for Cryptography).
; CHECK:   zbkc                             - 'Zbkc' (Carry-less multiply instructions for Cryptography).
; CHECK:   zbkx                             - 'Zbkx' (Crossbar permutation instructions).
; CHECK:   zbs                              - 'Zbs' (Single-Bit Instructions).
; CHECK:   zca                              - 'Zca' (part of the C extension, excluding compressed floating point loads/stores).
; CHECK:   zcb                              - 'Zcb' (Compressed basic bit manipulation instructions).
; CHECK:   zcd                              - 'Zcd' (Compressed Double-Precision Floating-Point Instructions).
; CHECK:   zce                              - 'Zce' (Compressed extensions for microcontrollers).
; CHECK:   zcf                              - 'Zcf' (Compressed Single-Precision Floating-Point Instructions).
; CHECK:   zcmop                            - 'Zcmop' (Compressed May-Be-Operations).
; CHECK:   zcmp                             - 'Zcmp' (sequenced instructions for code-size reduction).
; CHECK:   zcmt                             - 'Zcmt' (table jump instructions for code-size reduction).
; CHECK:   zdinx                            - 'Zdinx' (Double in Integer).
; CHECK:   zexth-fusion                     - Enable SLLI+SRLI to be fused to zero extension of halfword.
; CHECK:   zextw-fusion                     - Enable SLLI+SRLI to be fused to zero extension of word.
; CHECK:   zfa                              - 'Zfa' (Additional Floating-Point).
; CHECK:   zfbfmin                          - 'Zfbfmin' (Scalar BF16 Converts).
; CHECK:   zfh                              - 'Zfh' (Half-Precision Floating-Point).
; CHECK:   zfhmin                           - 'Zfhmin' (Half-Precision Floating-Point Minimal).
; CHECK:   zfinx                            - 'Zfinx' (Float in Integer).
; CHECK:   zhinx                            - 'Zhinx' (Half Float in Integer).
; CHECK:   zhinxmin                         - 'Zhinxmin' (Half Float in Integer Minimal).
; CHECK:   zic64b                           - 'Zic64b' (Cache Block Size Is 64 Bytes).
; CHECK:   zicbom                           - 'Zicbom' (Cache-Block Management Instructions).
; CHECK:   zicbop                           - 'Zicbop' (Cache-Block Prefetch Instructions).
; CHECK:   zicboz                           - 'Zicboz' (Cache-Block Zero Instructions).
; CHECK:   ziccamoa                         - 'Ziccamoa' (Main Memory Supports All Atomics in A).
; CHECK:   ziccif                           - 'Ziccif' (Main Memory Supports Instruction Fetch with Atomicity Requirement).
; CHECK:   zicclsm                          - 'Zicclsm' (Main Memory Supports Misaligned Loads/Stores).
; CHECK:   ziccrse                          - 'Ziccrse' (Main Memory Supports Forward Progress on LR/SC Sequences).
; CHECK:   zicntr                           - 'Zicntr' (Base Counters and Timers).
; CHECK:   zicond                           - 'Zicond' (Integer Conditional Operations).
; CHECK:   zicsr                            - 'Zicsr' (CSRs).
; CHECK:   zifencei                         - 'Zifencei' (fence.i).
; CHECK:   zihintntl                        - 'Zihintntl' (Non-Temporal Locality Hints).
; CHECK:   zihintpause                      - 'Zihintpause' (Pause Hint).
; CHECK:   zihpm                            - 'Zihpm' (Hardware Performance Counters).
; CHECK:   zimop                            - 'Zimop' (May-Be-Operations).
; CHECK:   zk                               - 'Zk' (Standard scalar cryptography extension).
; CHECK:   zkn                              - 'Zkn' (NIST Algorithm Suite).
; CHECK:   zknd                             - 'Zknd' (NIST Suite: AES Decryption).
; CHECK:   zkne                             - 'Zkne' (NIST Suite: AES Encryption).
; CHECK:   zknh                             - 'Zknh' (NIST Suite: Hash Function Instructions).
; CHECK:   zkr                              - 'Zkr' (Entropy Source Extension).
; CHECK:   zks                              - 'Zks' (ShangMi Algorithm Suite).
; CHECK:   zksed                            - 'Zksed' (ShangMi Suite: SM4 Block Cipher Instructions).
; CHECK:   zksh                             - 'Zksh' (ShangMi Suite: SM3 Hash Function Instructions).
; CHECK:   zkt                              - 'Zkt' (Data Independent Execution Latency).
; CHECK:   zmmul                            - 'Zmmul' (Integer Multiplication).
; CHECK:   ztso                             - 'Ztso' (Memory Model - Total Store Order).
; CHECK:   zvbb                             - 'Zvbb' (Vector basic bit-manipulation instructions).
; CHECK:   zvbc                             - 'Zvbc' (Vector Carryless Multiplication).
; CHECK:   zve32f                           - 'Zve32f' (Vector Extensions for Embedded Processors with maximal 32 EEW and F extension).
; CHECK:   zve32x                           - 'Zve32x' (Vector Extensions for Embedded Processors with maximal 32 EEW).
; CHECK:   zve64d                           - 'Zve64d' (Vector Extensions for Embedded Processors with maximal 64 EEW, F and D extension).
; CHECK:   zve64f                           - 'Zve64f' (Vector Extensions for Embedded Processors with maximal 64 EEW and F extension).
; CHECK:   zve64x                           - 'Zve64x' (Vector Extensions for Embedded Processors with maximal 64 EEW).
; CHECK:   zvfbfmin                         - 'Zvfbfmin' (Vector BF16 Converts).
; CHECK:   zvfbfwma                         - 'Zvfbfwma' (Vector BF16 widening mul-add).
; CHECK:   zvfh                             - 'Zvfh' (Vector Half-Precision Floating-Point).
; CHECK:   zvfhmin                          - 'Zvfhmin' (Vector Half-Precision Floating-Point Minimal).
; CHECK:   zvkb                             - 'Zvkb' (Vector Bit-manipulation used in Cryptography).
; CHECK:   zvkg                             - 'Zvkg' (Vector GCM instructions for Cryptography).
; CHECK:   zvkn                             - 'Zvkn' (shorthand for 'Zvkned', 'Zvknhb', 'Zvkb', and 'Zvkt').
; CHECK:   zvknc                            - 'Zvknc' (shorthand for 'Zvknc' and 'Zvbc').
; CHECK:   zvkned                           - 'Zvkned' (Vector AES Encryption & Decryption (Single Round)).
; CHECK:   zvkng                            - 'Zvkng' (shorthand for 'Zvkn' and 'Zvkg').
; CHECK:   zvknha                           - 'Zvknha' (Vector SHA-2 (SHA-256 only)).
; CHECK:   zvknhb                           - 'Zvknhb' (Vector SHA-2 (SHA-256 and SHA-512)).
; CHECK:   zvks                             - 'Zvks' (shorthand for 'Zvksed', 'Zvksh', 'Zvkb', and 'Zvkt').
; CHECK:   zvksc                            - 'Zvksc' (shorthand for 'Zvks' and 'Zvbc').
; CHECK:   zvksed                           - 'Zvksed' (SM4 Block Cipher Instructions).
; CHECK:   zvksg                            - 'Zvksg' (shorthand for 'Zvks' and 'Zvkg').
; CHECK:   zvksh                            - 'Zvksh' (SM3 Hash Function Instructions).
; CHECK:   zvkt                             - 'Zvkt' (Vector Data-Independent Execution Latency).
; CHECK:   zvl1024b                         - 'Zvl1024b' (Minimum Vector Length 1024).
; CHECK:   zvl128b                          - 'Zvl128b' (Minimum Vector Length 128).
; CHECK:   zvl16384b                        - 'Zvl16384b' (Minimum Vector Length 16384).
; CHECK:   zvl2048b                         - 'Zvl2048b' (Minimum Vector Length 2048).
; CHECK:   zvl256b                          - 'Zvl256b' (Minimum Vector Length 256).
; CHECK:   zvl32768b                        - 'Zvl32768b' (Minimum Vector Length 32768).
; CHECK:   zvl32b                           - 'Zvl32b' (Minimum Vector Length 32).
; CHECK:   zvl4096b                         - 'Zvl4096b' (Minimum Vector Length 4096).
; CHECK:   zvl512b                          - 'Zvl512b' (Minimum Vector Length 512).
; CHECK:   zvl64b                           - 'Zvl64b' (Minimum Vector Length 64).
; CHECK:   zvl65536b                        - 'Zvl65536b' (Minimum Vector Length 65536).
; CHECK:   zvl8192b                         - 'Zvl8192b' (Minimum Vector Length 8192).
