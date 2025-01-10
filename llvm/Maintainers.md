# LLVM Maintainers

This file is a list of the
[maintainers](https://llvm.org/docs/DeveloperPolicy.html#maintainers) for
LLVM.

## Current Maintainers

The following people are the active maintainers for the project. Please reach
out to them for code reviews, questions about their area of expertise, or other
assistance.

### Lead maintainer

The lead maintainer is responsible for all parts of LLVM not covered by somebody else.

Nikita Popov \
llvm@npopov.com, npopov@redhat.com (email), [nikic](https://github.com/nikic) (GitHub), nikic (Discourse)

### Transforms and analyses

#### AliasAnalysis

Nikita Popov \
llvm@npopov.com, npopov@redhat.com (email), [nikic](https://github.com/nikic) (GitHub), nikic (Discourse) \
Florian Hahn \
flo@fhahn.com (email), [fhahn](https://github.com/fhahn) (GitHub)

#### Attributor, OpenMPOpt

Johannes Doerfert \
jdoerfert@llnl.gov (email), [jdoerfert](https://github.com/jdoerfert) (GitHub)

#### ConstraintElimination

Florian Hahn \
flo@fhahn.com (email), [fhahn](https://github.com/fhahn) (GitHub)

#### InferAddressSpaces

Matt Arsenault \
Matthew.Arsenault@amd.com, arsenm2@gmail.com (email), [arsenm](https://github.com/arsenm) (GitHub)

#### Inlining

Arthur Eubanks \
aeubanks@google.com (email), [aeubanks](https://github.com/aeubanks) (GitHub) \
Mircea Trofin (esp. ML inliner) \
mtrofin@google.com (email), [mtrofin](https://github.com/mtrofin) (GitHub) \
Kazu Hirata (esp. module inliner and inline order) \
kazu@google.com (email), [kazutakahirata](https://github.com/kazutakahirata) (GitHub)

#### InstCombine, InstSimplify, ValueTracking, ConstantFold

Nikita Popov \
llvm@npopov.com, npopov@redhat.com (email), [nikic](https://github.com/nikic) (GitHub), nikic (Discourse) \
Yingwei Zheng \
dtcxzyw2333@gmail.com (email), [dtcxzyw](https://github.com/dtcxzyw) (GitHub)

#### InstrProfiling and related parts of ProfileData

Justin Bogner \
mail@justinbogner.com (email), [bogner](https://github.com/bogner) (GitHub)

#### SampleProfile and related parts of ProfileData

Diego Novillo \
dnovillo@google.com (email), [dnovillo](https://github.com/dnovillo) (GitHub)

#### New pass manager, CGSCC, LazyCallGraph

Arthur Eubanks \
aeubanks@google.com (email), [aeubanks](https://github.com/aeubanks) (GitHub)

#### LoopStrengthReduce

Quentin Colombet \
quentin.colombet@gmail.com (email), [qcolombet](https://github.com/qcolombet) (GitHub)

#### LoopVectorize

Florian Hahn \
flo@fhahn.com (email), [fhahn](https://github.com/fhahn) (GitHub)

#### MemorySSA

Alina Sbirlea \
asbirlea@google.com (email), [alinas](https://github.com/alinas) (GitHub)

#### SandboxVectorizer

Vasileios Porpodas \
vporpodas@google.com (email), [vporpo](https://github.com/vporpo) (GitHub) \
Jorge Gorbe Moya \
jgorbe@google.com (email), [slackito](https://github.com/slackito) (GitHub)

#### ScalarEvolution, IndVarSimplify

Philip Reames \
listmail@philipreames.com (email), [preames](https://github.com/preames) (GitHub)

#### SLPVectorizer

Alexey Bataev \
a.bataev@outlook.com (email), [alexey-bataev](https://github.com/alexey-bataev) (GitHub)

#### SROA, Mem2Reg

Chandler Carruth \
chandlerc@gmail.com, chandlerc@google.com (email), [chandlerc](https://github.com/chandlerc) (GitHub)

### Instrumentation and sanitizers

#### Sanitizers not covered by someone else

Vitaly Buka \
vitalybuka@google.com (email), [vitalybuka](https://github.com/vitalybuka) (GitHub)

#### NumericalStabilitySanitizer

Alexander Shaposhnikov \
ashaposhnikov@google.com (email), [alexander-shaposhnikov](https://github.com/alexander-shaposhnikov) (GitHub)

#### RealtimeSanitizer

Christopher Apple \
cja-private@pm.me (email), [cjappl](https://github.com/cjappl) (GitHub) \
David Trevelyan \
david.trevelyan@gmail.com (email), [davidtrevelyan](https://github.com/davidtrevelyan) (GitHub)

### Generic backend and code generation

#### Parts of code generator not covered by someone else

Matt Arsenault \
Matthew.Arsenault@amd.com, arsenm2@gmail.com (email), [arsenm](https://github.com/arsenm) (GitHub)

#### SelectionDAG

Simon Pilgrim \
llvm-dev@redking.me.uk (email), [RKSimon](https://github.com/RKSimon) (GitHub) \
Craig Topper \
craig.topper@sifive.com (email), [topperc](https://github.com/topperc) (GitHub)

#### Instruction scheduling

Matthias Braun \
matze@braunis.de (email), [MatzeB](https://github.com/MatzeB) (GitHub)

#### VLIW Instruction Scheduling, Packetization

Sergei Larin \
slarin@codeaurora.org (email)

#### Register allocation

Quentin Colombet \
quentin.colombet@gmail.com (email), [qcolombet](https://github.com/qcolombet) (GitHub)

#### MC layer

Fangrui Song \
i@maskray.me (email), [MaskRay](https://github.com/MaskRay) (GitHub)

#### Windows ABI and codegen

Reid Kleckner \
rnk@google.com (email), [rnk](https://github.com/rnk) (GitHub)

### Backends / Targets

#### AArch64 backend

Tim Northover \
t.p.northover@gmail.com (email), [TNorthover](https://github.com/TNorthover) (GitHub)

#### AMDGPU backend

Matt Arsenault \
Matthew.Arsenault@amd.com, arsenm2@gmail.com (email), [arsenm](https://github.com/arsenm) (GitHub)

#### ARC backend

Mark Schimmel \
marksl@synopsys.com (email), [markschimmel](https://github.com/markschimmel) (GitHub)

#### ARM backend

David Green \
david.green@arm.com (email), [davemgreen](https://github.com/davemgreen) (GitHub) \
Oliver Stannard (Especially assembly/dissassembly) \
oliver.stannard@arm.com (email), [ostannard](https://github.com/ostannard) (GitHub) \
Nashe Mncube \
nashe.mncube@arm.com (email), [nasherm](https://github.com/nasherm) (GitHub) \
Peter Smith (Anything ABI) \
peter.smith@arm.com (email), [smithp35](https://github.com/smithp35) (GitHub) \
Ties Stuij (GlobalISel and early arch support) \
ties.stuij@arm.com (email), [stuij](https://github.com/stuij) (GitHub)

#### AVR backend

Ben Shi \
2283975856@qq.com, powerman1st@163.com (email), [benshi001](https://github.com/benshi001) (GitHub)

#### BPF backend

Yonghong Song \
yhs@fb.com (email), [yonghong-song](https://github.com/yonghong-song) (GitHub) \
Eduard Zingerman \
eddyz87@gmail.com (email), [eddyz87](https://github.com/eddyz87) (GitHub)

#### CSKY backend

Zi Xuan Wu (Zeson) \
zixuan.wu@linux.alibaba.com (email), [zixuan-wu](https://github.com/zixuan-wu) (GitHub)

#### DirectX backend

Justin Bogner \
mail@justinbogner.com (email), [bogner](https://github.com/bogner) (GitHub)

#### Hexagon backend

Sundeep Kushwaha \
sundeepk@quicinc.com (email), [SundeepKushwaha](https://github.com/SundeepKushwaha) (GitHub)

#### Lanai backend

Jacques Pienaar \
jpienaar@google.com (email), [jpienaar](https://github.com/jpienaar) (GitHub)

#### LoongArch backend

Weining Lu \
luweining@loongson.cn (email), [SixWeining](https://github.com/SixWeining) (GitHub)

#### M68k backend

Min-Yih Hsu \
min@myhsu.dev (email), [mshockwave](https://github.com/mshockwave) (GitHub)

#### MSP430 backend

Anton Korobeynikov \
anton@korobeynikov.info (email), [asl](https://github.com/asl) (GitHub)

#### NVPTX backend

Justin Holewinski \
jholewinski@nvidia.com (email), [jholewinski](https://github.com/jholewinski) (GitHub) \
Artem Belevich \
tra@google.com (email), [Artem-B](https://github.com/Artem-B) (GitHub) \
Alex MacLean \
amaclean@nvidia.com (email), [AlexMaclean](https://github.com/AlexMaclean) (GitHub) \
Justin Fargnoli \
jfargnoli@nvidia.com (email), [justinfargnoli](https://github.com/justinfargnoli) (GitHub)

#### PowerPC backend

Zheng Chen \
czhengsz@cn.ibm.com (email), [chenzheng1030](https://github.com/chenzheng1030) (GitHub)

#### RISCV backend

Alex Bradbury \
asb@igalia.com (email), [asb](https://github.com/asb) (GitHub) \
Craig Topper \
craig.topper@sifive.com (email), [topperc](https://github.com/topperc) (GitHub) \
Philip Reames \
listmail@philipreames.com (email), [preames](https://github.com/preames) (GitHub)

#### Sparc backend

Koakuma \
koachan@protonmail.com (email), [koachan](https://github.com/koachan) (GitHub)

#### SPIRV backend

Ilia Diachkov \
ilia.diachkov@gmail.com (email), [iliya-diyachkov](https://github.com/iliya-diyachkov) (GitHub)

#### SystemZ backend

Ulrich Weigand \
uweigand@de.ibm.com (email), [uweigand](https://github.com/uweigand) (GitHub)

#### VE backend

Kazushi Marukawa \
marukawa@nec.com (email), [kaz7](https://github.com/kaz7) (GitHub)

#### WebAssembly backend

Dan Gohman \
llvm@sunfishcode.online (email), [sunfishcode](https://github.com/sunfishcode) (GitHub)

#### X86 backend

Simon Pilgrim \
llvm-dev@redking.me.uk (email), [RKSimon](https://github.com/RKSimon) (GitHub) \
Phoebe Wang \
phoebe.wang@intel.com (email), [phoebewang](https://github.com/phoebewang) (GitHub)

#### XCore backend

Nigel Perks \
nigelp@xmos.com (email), [nigelp-xmos](https://github.com/nigelp-xmos) (GitHub)

#### Xtensa backend

Andrei Safronov \
andrei.safronov@espressif.com (email), [andreisfr](https://github.com/andreisfr) (GitHub)

### Libraries and shared infrastructure

#### ADT, Support

David Blaikie \
dblaikie@gmail.com (email), [dwblaikie](https://github.com/dwblaike) (GitHub) \
Jakub Kuderski \
jakub@nod-labs.com (email), [kuhar](https://github.com/kuhar) (GitHub)

#### Bitcode

Peter Collingbourne \
peter@pcc.me.uk (email), [pcc](https://github.com/pcc) (GitHub)

#### CMake and library layering

Chandler Carruth \
chandlerc@gmail.com, chandlerc@google.com (email), [chandlerc](https://github.com/chandlerc) (GitHub)

#### Debug info and DWARF

Adrian Prantl \
aprantl@apple.com (email), [adrian-prantl](https://github.com/adrian-prantl) (GitHub) \
David Blaikie (especially type information) \
dblaikie@gmail.com (email), [dwblaikie](https://github.com/dwblaike) (GitHub) \
Jeremy Morse (especially variable information) \
jeremy.morse@sony.com (email), [jmorse](https://github.com/jmorse) (GitHub) \
Jonas Devlieghere (especially dsymutil/DWARFLinker) \
jonas@devlieghere.com (email), [JDevlieghere](https://github.com/JDevlieghere) (GitHub) \
Eric Christopher \
echristo@gmail.com (email), [echristo](https://github.com/echristo) (GitHub)

#### IR Linker and LTO

Teresa Johnson \
tejohnson@google.com (email), [teresajohnson](https://github.com/teresajohnson) (GitHub)

#### MCJIT, Orc, RuntimeDyld, PerfJITEvents

Lang Hames \
lhames@gmail.com (email), [lhames](https://github.com/lhames) (GitHub)

#### SandboxIR

Vasileios Porpodas \
vporpodas@google.com (email), [vporpo](https://github.com/vporpo) (GitHub) \
Jorge Gorbe Moya \
jgorbe@google.com (email), [slackito](https://github.com/slackito) (GitHub)

#### TableGen

Rahul Joshi \
rjoshi@nvidia.com (email), [jurahul](https://github.com/jurahul) (GitHub)

#### TextAPI

Cyndy Ishida \
cyndyishida@gmail.com (email), [cyndyishida](https://github.com/cyndyishida) (GitHub)

### Tools

#### llvm-mca and MCA library

Andrea Di Biagio \
andrea.dibiagio@sony.com, andrea.dibiagio@gmail.com (email), [adibiagio](https://github.com/adibiagio) (GitHub)

#### Binary Utilities

James Henderson \
james.henderson@sony.com (email), [jh7370](https://github.com/jh7370) (GitHub) \
Fangrui Song \
i@maskray.me (email), [MaskRay](https://github.com/MaskRay) (GitHub)

#### Gold plugin

Teresa Johnson \
tejohnson@google.com (email), [teresajohnson](https://github.com/teresajohnson) (GitHub)

### Other

#### Release management

Odd releases:

Tobias Hieta \
tobias@hieta.se (email), [tru](https://github.com/tru) (GitHub)

Even releases:

Tom Stellard \
tstellar@redhat.com (email), [tstellar](https://github.com/tstellar) (GitHub)

#### MinGW support

Martin Storsjö \
martin@martin.st (email), [mstorsjo](https://github.com/mstorsjo) (GitHub)

#### Sony PlayStation support

Jeremy Morse \
jeremy.morse@sony.com (email), [jmorse](https://github.com/jmorse) (GitHub)

#### Inline assembly

Eric Christopher \
echristo@gmail.com (email), [echristo](https://github.com/echristo) (GitHub)

#### Exception handling

Anton Korobeynikov \
anton@korobeynikov.info (email), [asl](https://github.com/asl) (GitHub)

#### LLVM Buildbot

Galina Kistanova \
gkistanova@gmail.com (email), [gkistanova](https://github.com/gkistanova) (GitHub)

### Other subprojects

Some subprojects maintain their own list of per-component maintainers.

[Bolt maintainers](https://github.com/llvm/llvm-project/blob/main/bolt/Maintainers.txt)

[Clang maintainers](https://github.com/llvm/llvm-project/blob/main/clang/Maintainers.rst)

[Clang-tools-extra maintainers](https://github.com/llvm/llvm-project/blob/main/clang-tools-extra/Maintainers.txt)

[Compiler-rt maintainers](https://github.com/llvm/llvm-project/blob/main/compiler-rt/Maintainers.md)

[Flang maintainers](https://github.com/llvm/llvm-project/blob/main/flang/Maintainers.txt)

[libc++ maintainers](https://github.com/llvm/llvm-project/blob/main/libcxx/Maintainers.md)

[libclc maintainers](https://github.com/llvm/llvm-project/blob/main/libclc/Maintainers.md)

[LLD maintainers](https://github.com/llvm/llvm-project/blob/main/lld/Maintainers.md)

[LLDB maintainers](https://github.com/llvm/llvm-project/blob/main/lldb/Maintainers.rst)

[LLVM OpenMP Library maintainers](https://github.com/llvm/llvm-project/blob/main/openmp/Maintainers.md)

[Polly maintainers](https://github.com/llvm/llvm-project/blob/main/polly/Maintainers.md)

## Inactive Maintainers

The following people have graciously spent time performing maintainer
responsibilities but are no longer active in that role. Thank you for all your
help with the success of the project!

### Emeritus lead maintainers

Chris Lattner \
sabre@nondot.org (email), [lattner](https://github.com/lattner) (GitHub), clattner (Discourse)

### Inactive or former component maintainers

Paul C. Anagnostopoulos (paul@windfall.com, [Paul-C-Anagnostopoulos](https://github.com/Paul-C-Anagnostopoulos)) -- TableGen \
Justin Bogner (mail@justinbogner.com, [bogner](https://github.com/bogner)) -- SelectionDAG \
Chandler Carruth (chandlerc@gmail.com, chandlerc@google.com, [chandlerc](https://github.com/chandlerc)) -- ADT, Support, Inlining \
Peter Collingbourne (peter@pcc.me.uk, [pcc](https://github.com/pcc)) -- LTO \
Evan Cheng (evan.cheng@apple.com) -- Parts of code generator not covered by someone else \
Jake Ehrlich (jakehehrlich@google.com, [jakehehrlich](https://github.com/jakehehrlich)) -- llvm-objcopy and ObjCopy library \
Hal Finkel (hfinkel@anl.gov, [hfinkel](https://github.com/hfinkel) -- AliasAnalysis \
Renato Golin (rengolin@systemcall.eu, [rengolin](https://github.com/rengolin)) -- ARM backend \
Venkatraman Govindaraju (venkatra@cs.wisc.edu, [vegovin](https://github.com/vegovin) -- Sparc backend \
James Grosbach (grosbach@apple.com) -- MC layer \
Anton Korobeynikov (anton@korobeynikov.info, [asl](https://github.com/asl)) -- ARM EABI, Windows codegen \
Benjamin Kramer (benny.kra@gmail.com, [d0k](https://github.com/d0k)) -- DWARF Parser \
David Majnemer (david.majnemer@gmail.com, [majnemer](https://github.com/majnemer)) -- InstCombine, ConstantFold \
Chad Rosier (mcrosier@codeaurora.org) -- FastISel \
Hans Wennborg (hans@chromium.org, [zmodem](https://github.com/zmodem)) -- Release management \
Kostya Serebryany ([kcc](https://github.com/kcc)) -- Sanitizers \
Michael Spencer (bigcheesegs@gmail.com), [Bigcheese](https://github.com/Bigcheese)) -- Windows support in object tools \
Alexei Starovoitov (alexei.starovoitov@gmail.com, [4ast](https://github.com/4ast)) -- BPF backend \
Evgeniy Stepanov ([eugenis](https://github.com/eugenis)) -- Sanitizers

### Former maintainers of removed components

Duncan Sands (baldrick@free.fr, [CunningBaldrick](https://github.com/CunningBaldrick)) -- DragonEgg \
Hal Finkel (hfinkel@anl.gov, [hfinkel](https://github.com/hfinkel)) -- LoopReroll

