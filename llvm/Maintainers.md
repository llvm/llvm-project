# LLVM Maintainers

This file is a list of the
[maintainers](https://llvm.org/docs/DeveloperPolicy.html#maintainers) for
LLVM.

## Current Maintainers

The following people are the active maintainers for the project. Please reach
out to them for code reviews, questions about their area of expertise, or other
assistance.

**Warning: The maintainer list for LLVM is currently not up to date.**

### Lead maintainer

The lead maintainer is responsible for all parts of LLVM not covered by somebody else.

Nikita Popov \
llvm@npopov.com, npopov@redhat.com (email), [nikic](https://github.com/nikic) (GitHub), nikic (Discourse)

### Transforms and analyses

#### AliasAnalysis

Hal Finkel \
hfinkel@anl.gov (email), [hfinkel](https://github.com/hfinkel) (GitHub)

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

Chandler Carruth \
chandlerc@gmail.com, chandlerc@google.com (email), [chandlerc](https://github.com/chandlerc) (GitHub)

#### InstCombine, ConstantFold

David Majnemer \
david.majnemer@gmail.com (email), [majnemer](https://github.com/majnemer) (GitHub)

#### InstrProfiling and related parts of ProfileData

Justin Bogner \
mail@justinbogner.com (email), [bogner](https://github.com/bogner) (GitHub)

#### SampleProfile and related parts of ProfileData

Diego Novillo \
dnovillo@google.com (email), [dnovillo](https://github.com/dnovillo) (GitHub)

#### LoopStrengthReduce

Quentin Colombet \
quentin.colombet@gmail.com (email), [qcolombet](https://github.com/qcolombet) (GitHub)

#### LoopVectorize

Florian Hahn \
flo@fhahn.com (email), [fhahn](https://github.com/fhahn) (GitHub)

#### SandboxVectorizer

Vasileios Porpodas \
vporpodas@google.com (email), [vporpo](https://github.com/vporpo) (GitHub)
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

#### AddressSanitizer, ThreadSanitizer

Kostya Serebryany \
kcc@google.com (email), [kcc](https://github.com/kcc) (GitHub)

#### MemorySanitizer

Evgeniy Stepanov \
eugenis@google.com (email), [eugenis](https://github.com/eugenis) (GitHub)

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

#### FastISel

Chad Rosier \
mcrosier@codeaurora.org (email)

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

James Grosbach \
grosbach@apple.com (email)

#### Windows codegen

Anton Korobeynikov \
anton@korobeynikov.info (email), [asl](https://github.com/asl) (GitHub)

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

Alexei Starovoitov \
alexei.starovoitov@gmail.com (email), [4ast](https://github.com/4ast) (GitHub)

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
jholewinski@nvidia.com (email), [jholewinski](https://github.com/jholewinski) (GitHub)
Artem Belevich \
tra@google.com (email), [Artem-B](https://github.com/Artem-B) (GitHub)
Alex MacLean \
amaclean@nvidia.com (email), [AlexMaclean](https://github.com/AlexMaclean) (GitHub)
Justin Fargnoli \
jfargnoli@nvidia.com (email), [justinfargnoli](https://github.com/justinfargnoli) (GitHub)

#### PowerPC backend

Zheng Chen \
czhengsz@cn.ibm.com (email), [chenzheng1030](https://github.com/chenzheng1030) (GitHub)

#### RISCV backend

Alex Bradbury \
asb@igalia.com (email), [asb](https://github.com/asb) (GitHub)

#### Sparc backend

Venkatraman Govindaraju \
venkatra@cs.wisc.edu (email), [vegovin](https://github.com/vegovin) (GitHub)

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

Chandler Carruth \
chandlerc@gmail.com, chandlerc@google.com (email), [chandlerc](https://github.com/chandlerc) (GitHub)

#### Bitcode

Peter Collingbourne \
peter@pcc.me.uk (email), [pcc](https://github.com/pcc) (GitHub)

#### CMake and library layering

Chandler Carruth \
chandlerc@gmail.com, chandlerc@google.com (email), [chandlerc](https://github.com/chandlerc) (GitHub)

#### Debug info

Eric Christopher \
echristo@gmail.com (email), [echristo](https://github.com/echristo) (GitHub)

#### DWARF Parser

Benjamin Kramer \
benny.kra@gmail.com (email), [d0k](https://github.com/d0k) (GitHub)

#### IR Linker

Teresa Johnson \
tejohnson@google.com (email), [teresajohnson](https://github.com/teresajohnson) (GitHub)

#### LTO

Peter Collingbourne \
peter@pcc.me.uk (email), [pcc](https://github.com/pcc) (GitHub)

#### MCJIT, Orc, RuntimeDyld, PerfJITEvents

Lang Hames \
lhames@gmail.com (email), [lhames](https://github.com/lhames) (GitHub)

#### SandboxIR

Vasileios Porpodas \
vporpodas@google.com (email), [vporpo](https://github.com/vporpo) (GitHub)
Jorge Gorbe Moya \
jgorbe@google.com (email), [slackito](https://github.com/slackito) (GitHub)

#### TableGen

Paul C. Anagnostopoulos \
paul@windfall.com (email)

#### TextAPI

Cyndy Ishida \
cyndyishida@gmail.com (email), [cyndyishida](https://github.com/cyndyishida) (GitHub)

### Tools

#### llvm-mca and MCA library

Andrea Di Biagio \
andrea.dibiagio@sony.com, andrea.dibiagio@gmail.com (email), [adibiagio](https://github.com/adibiagio) (GitHub)

#### llvm-objcopy and ObjCopy library

Jake Ehrlich \
jakehehrlich@google.com (email), [jakehehrlich](https://github.com/jakehehrlich) (GitHub)

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
martin@martin.st (email), [mstrorsjo](https://github.com/mstrorsjo) (GitHub)

#### Windows support in object tools

Michael Spencer \
bigcheesegs@gmail.com (email), [Bigcheese](https://github.com/Bigcheese) (GitHub)

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
Others only have a lead maintainer listed here.

[Bolt maintainers](https://github.com/llvm/llvm-project/blob/main/bolt/CODE_OWNERS.TXT)

[Clang maintainers](https://github.com/llvm/llvm-project/blob/main/clang/Maintainers.rst)

[Clang-tools-extra maintainers](https://github.com/llvm/llvm-project/blob/main/clang-tools-extra/CODE_OWNERS.TXT)

[Compiler-rt maintainers](https://github.com/llvm/llvm-project/blob/main/compiler-rt/CODE_OWNERS.TXT)

[Flang maintainers](https://github.com/llvm/llvm-project/blob/main/flang/Maintainers.txt)

[LLD maintainers](https://github.com/llvm/llvm-project/blob/main/lld/CODE_OWNERS.TXT)

[LLDB maintainers](https://github.com/llvm/llvm-project/blob/main/lldb/Maintainers.rst)

#### libc++

Louis Dionne \
ldionne.2@gmail.com (email), [ldionne](https://github.com/ldionne) (GitHub)

#### libclc

Tom Stellard \
tstellar@redhat.com (email), [tstellar](https://github.com/tstellar) (GitHub)

#### OpenMP (runtime library)

Andrey Churbanov \
andrey.churbanov@intel.com (email), [AndreyChurbanov](https://github.com/AndreyChurbanov) (GitHub)

#### Polly

Tobias Grosser \
tobias@grosser.es (email), [tobiasgrosser](https://github.com/tobiasgrosser) (GitHub)

## Inactive Maintainers

The following people have graciously spent time performing maintainer
responsibilities but are no longer active in that role. Thank you for all your
help with the success of the project!

### Emeritus lead maintainers

Chris Lattner \
sabre@nondot.org (email), [lattner](https://github.com/lattner) (GitHub), clattner (Discourse)

### Inactive or former component maintainers

Justin Bogner (mail@justinbogner.com, [bogner](https://github.com/bogner)) -- SelectionDAG \
Evan Cheng (evan.cheng@apple.com) -- Parts of code generator not covered by someone else \
Renato Golin (rengolin@systemcall.eu, [rengolin](https://github.com/rengolin)) -- ARM backend \
Anton Korobeynikov (anton@korobeynikov.info, [asl](https://github.com/asl)) -- ARM EABI \
Hans Wennborg (hans@chromium.org, [zmodem](https://github.com/zmodem)) -- Release management \

### Former maintainers of removed components

Duncan Sands (baldrick@free.fr, [CunningBaldrick](https://github.com/CunningBaldrick)) -- DragonEgg \
Hal Finkel (hfinkel@anl.gov, [hfinkel](https://github.com/hfinkel)) -- LoopReroll

