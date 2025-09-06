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

#### HashRecognize

Ramkumar Ramachandra \
r@artagnon.com (email), [artagnon](https://github.com/artagnon) (GitHub), artagnon (Discourse) \
Piotr Fusik \
p.fusik@samsung.com (email), [pfusik](https://github.com/pfusik) (GitHub)

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

#### LoopInterchange

Madhur Amilkanthwar \
madhura@nvidia.com (email), [madhur13490](https://github.com/madhur13490) (GitHub)

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

#### ARM and AArch64 backends

David Green \
david.green@arm.com (email), [davemgreen](https://github.com/davemgreen) (GitHub) \
Amara Emerson (esp. AArch64 GlobalISel) \
amara@apple.com (email), [aemerson](https://github.com/aemerson) (GitHub) \
Eli Friedman (esp. ARM64EC) \
efriedma@quicinc.com (email), [efriedma-quic](https://github.com/efriedma-quic) (GitHub) \
Sjoerd Meijer \
smeijer@nvidia.com (email), [sjoerdmeijer](https://github.com/sjoerdmeijer) (GitHub) \
Nashe Mncube \
nashe.mncube@arm.com (email), [nasherm](https://github.com/nasherm) (GitHub) \
Sander de Smalen (esp. scalable vectorization/SVE/SME) \
sander.desmalen@arm.com (email), [sdesmalen-arm](https://github.com/sdesmalen-arm) (GitHub) \
Peter Smith (Anything ABI) \
peter.smith@arm.com (email), [smithp35](https://github.com/smithp35) (GitHub) \
Oliver Stannard (esp. assembly/dissassembly) \
oliver.stannard@arm.com (email), [ostannard](https://github.com/ostannard) (GitHub) \
Ties Stuij (Arm GlobalISel and early arch support) \
ties.stuij@arm.com (email), [stuij](https://github.com/stuij) (GitHub)

#### AMDGPU backend

Matt Arsenault \
Matthew.Arsenault@amd.com, arsenm2@gmail.com (email), [arsenm](https://github.com/arsenm) (GitHub)

#### ARC backend

Mark Schimmel \
marksl@synopsys.com (email), [markschimmel](https://github.com/markschimmel) (GitHub)

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

#### PowerPC backend

Amy Kwan (esp. release issues) \
Amy.Kwan1@ibm.com (email), [amy-kwan](https://github.com/amy-kwan) (GitHub) \
Lei Huang \
lei@ca.ibm.com (email), [lei137](https://github.com/lei137) (GitHub) \
Sean Fertile (esp. ABI/ELF/XCOFF) \
sfertile@ca.ibm.com (email), [mandlebug](https://github.com/mandlebug) (GitHub) \
Zhijian Lin \
zhijian@ca.ibm.com (email), [diggerlin](https://github.com/diggerlin) (GitHub) \
Maryam Moghadas \
maryammo@ca.ibm.com (email), [maryammo](https://github.com/maryammo) (GitHub) \
Roland Froese \
froese@ca.ibm.com (email), [RolandF77](https://github.com/RolandF77) (GitHub) \
llvmonpower \
powerllvm@ca.ibm.com (email), [llvmonpower](https://github.com/llvmonpower) (GitHub)

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

Vyacheslav Levytskyy \
vyacheslav.levytskyy@intel.com, vyacheslav.levytskyy@gmail.com (email), [VyacheslavLevytskyy](https://github.com/VyacheslavLevytskyy) (GitHub)

Nathan Gauër \
brioche@google.com (email), [Keenuts](https://github.com/Keenuts) (GitHub)

#### SystemZ backend

Ulrich Weigand \
uweigand@de.ibm.com (email), [uweigand](https://github.com/uweigand) (GitHub)

#### VE backend

Kazushi Marukawa \
marukawa@nec.com (email), [kaz7](https://github.com/kaz7) (GitHub)

#### WebAssembly backend

Derek Schuff \
dschuff@chromium.org (email), [dschuff](https://github.com/dschuff) (GitHub) \
Heejin Ahn \
aheejin@gmail.com (email), [aheejin](https://github.com/aheejin) (GitHub)

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

#### CMake

Petr Hosek \
phosek@google.com (email), [petrhosek](https://github.com/petrhosek) (GitHub)

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

#### Library layering

Takumi Nakamura \
geek4civic@gmail.com (email), [chapuni](https://github.com/chapuni) (GitHub)

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
andrea.dibiagio@sony.com, andrea.dibiagio@gmail.com (email), [adibiagio](https://github.com/adibiagio) (GitHub) \
Min-Yih Hsu \
min.hsu@sifive.com, min@myhsu.dev (email), [mshockwave](https://github.com/mshockwave) (GitHub)

#### llvm-cov and Coverage parts of ProfileData

Takumi Nakamura \
geek4civic@gmail.com (email), [chapuni](https://github.com/chapuni) (GitHub) \
Alan Phipps \
a-phipps@ti.com (email), [evodius96](https://github.com/evodius96) (GitHub)

#### Binary Utilities

James Henderson \
james.henderson@sony.com (email), [jh7370](https://github.com/jh7370) (GitHub) \
Fangrui Song \
i@maskray.me (email), [MaskRay](https://github.com/MaskRay) (GitHub)

#### Gold plugin

Teresa Johnson \
tejohnson@google.com (email), [teresajohnson](https://github.com/teresajohnson) (GitHub)

#### llvm-exegesis

Aiden Grossman \
agrossman154@yahoo.com (email), [boomanaiden154](https://github.com/boomanaiden154) (Github)

#### llvm-reduce

Matt Arsenault \
Matthew.Arsenault@amd.com, arsenm2@gmail.com (email), [arsenm](https://github.com/arsenm) (GitHub)

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

Reid Kleckner \
rnk@google.com (email), [rnk](https://github.com/rnk) (GitHub)

#### LLVM Buildbot

Galina Kistanova \
gkistanova@gmail.com (email), [gkistanova](https://github.com/gkistanova) (GitHub)

### Other subprojects

Some subprojects maintain their own list of per-component maintainers.

[Bolt maintainers](https://github.com/llvm/llvm-project/blob/main/bolt/Maintainers.txt)

[Clang maintainers](https://github.com/llvm/llvm-project/blob/main/clang/Maintainers.rst)

[Clang-tools-extra maintainers](https://github.com/llvm/llvm-project/blob/main/clang-tools-extra/Maintainers.txt)

[Compiler-rt maintainers](https://github.com/llvm/llvm-project/blob/main/compiler-rt/Maintainers.md)

[Flang maintainers](https://github.com/llvm/llvm-project/blob/main/flang/Maintainers.md)

[libc++ maintainers](https://github.com/llvm/llvm-project/blob/main/libcxx/Maintainers.md)

[Libc maintainers](https://github.com/llvm/llvm-project/blob/main/libc/Maintainers.rst)

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
Chandler Carruth (chandlerc@gmail.com, chandlerc@google.com, [chandlerc](https://github.com/chandlerc)) -- ADT, Support, Inlining, CMake and library layering \
Peter Collingbourne (peter@pcc.me.uk, [pcc](https://github.com/pcc)) -- LTO \
Evan Cheng (evan.cheng@apple.com) -- Parts of code generator not covered by someone else \
Jake Ehrlich (jakehehrlich@google.com, [jakehehrlich](https://github.com/jakehehrlich)) -- llvm-objcopy and ObjCopy library \
Hal Finkel (hfinkel@anl.gov, [hfinkel](https://github.com/hfinkel) -- AliasAnalysis \
Justin Fargnoli (jfargnoli@nvidia.com, [justinfargnoli](https://github.com/justinfargnoli)) -- NVPTX backend \
Renato Golin (rengolin@systemcall.eu, [rengolin](https://github.com/rengolin)) -- ARM backend \
Venkatraman Govindaraju (venkatra@cs.wisc.edu, [vegovin](https://github.com/vegovin) -- Sparc backend \
James Grosbach (grosbach@apple.com) -- MC layer \
Anton Korobeynikov (anton@korobeynikov.info, [asl](https://github.com/asl)) -- ARM EABI, Windows codegen, Exception handling \
Benjamin Kramer (benny.kra@gmail.com, [d0k](https://github.com/d0k)) -- DWARF Parser \
David Majnemer (david.majnemer@gmail.com, [majnemer](https://github.com/majnemer)) -- InstCombine, ConstantFold \
Tim Northover (t.p.northover@gmail.com, [TNorthover](https://github.com/TNorthover)) -- AArch64 backend \
Chad Rosier (mcrosier@codeaurora.org) -- FastISel \
Hans Wennborg (hans@chromium.org, [zmodem](https://github.com/zmodem)) -- Release management \
Kostya Serebryany ([kcc](https://github.com/kcc)) -- Sanitizers \
Michael Spencer (bigcheesegs@gmail.com), [Bigcheese](https://github.com/Bigcheese)) -- Windows support in object tools \
Alexei Starovoitov (alexei.starovoitov@gmail.com, [4ast](https://github.com/4ast)) -- BPF backend \
Evgeniy Stepanov ([eugenis](https://github.com/eugenis)) -- Sanitizers \
Zheng Chen (czhengsz@cn.ibm.com, [chenzheng1030](https://github.com/chenzheng1030)) -- PowerPC backend \
Dan Gohman (llvm@sunfishcode.online, [sunfishcode](https://github.com/sunfishcode)) -- WebAssembly backend

### Former maintainers of removed components

Duncan Sands (baldrick@free.fr, [CunningBaldrick](https://github.com/CunningBaldrick)) -- DragonEgg \
Hal Finkel (hfinkel@anl.gov, [hfinkel](https://github.com/hfinkel)) -- LoopReroll
