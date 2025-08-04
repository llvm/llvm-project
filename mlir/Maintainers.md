# MLIR Maintainers

This file is a list of the
[maintainers](https://llvm.org/docs/DeveloperPolicy.html#maintainers) for MLIR.

The following people are the active maintainers for the project. For the sake of
simplicity, responsibility areas are subdivided into broad categories, which are
further subdivided into individual components, such as dialects. Please reach
out to them for code reviews, questions about their area of expertise, or other
assistance.

## Core

Core components of MLIR, including core IR, analyses and rewriters, fundamental
dialects, build system and language bindings.

- Alex Zinenko \
  ftynse@gmail.com (email),
  [@ftynse](https://github.com/ftynse) (GitHub),
  ftynse (Discourse)
- Jacques Pienaar \
  jpienaar@google.com (email),
  [@jpienaar](https://github.com/jpienaar) (GitHub),
  jpienaar (Discourse)
- Mehdi Amini \
  joker.eph@gmail.com (email),
  [@joker-eph](https://github.com/joker-eph) (GitHub),
  mehdi_amini (Discourse)

## Egress

MLIR components pertaining to egress flows from MLIR, in particular to LLVM IR.

- Matthias Springer \
  me@m-sp.org (email),
  [@matthias-springer](https://github.com/matthias-springer) (GitHub),
  matthias-springer (Discourse)
- Andrzej Warzynski \
  andrzej.warzynski@arm.com (email),
  [@banach-space](https://github.com/banach-space) (GitHub),
  banach-space (Discourse)
- Tobias Gysi \
  tobias.gysi@nextsilicon.com (email),
  [@gysit](https://github.com/gysit) (GitHub),
  gysit (Discourse)

### Dialects

The `egress` maintainer refers to the people working in the Egress category,
with the point-of-contact being the maintainers above. Named maintainers, if
available, should be contacted first, as they're more active in those areas.

#### Lowering Dialects
* ‘llvm’ Dialect (egress)
* ‘SPIR-V’ Dialect ([@kuhar](https://github.com/kuhar), [@antiagainst](https://github.com/antiagainst))
* ‘emitc’ Dialect ([@aniragil](https://github.com/aniragil), [@marbre](https://github.com/marbre))

#### GPU Dialects
* ‘gpu’ Dialect ([@fabianmcg](https://github.com/fabianmcg))
* ‘amdgpu’ Dialect ([@krzysz00](https://github.com/krzysz00))
* ‘rocdl’ Dialect ([@krzysz00](https://github.com/krzysz00))
* ‘nvgpu’ Dialect ([@grypp](https://github.com/grypp))
* ‘nvvm’ Dialect ([@grypp](https://github.com/grypp))
* ‘xegpu’ Dialect ([@chencha3](https://github.com/chencha3), [@Jianhui-Li](https://github.com/Jianhui-Li))
* 'xevm' Dialect ([@silee2](https://github.com/silee2))

#### CPU Dialects
* ‘arm_neon’ Dialect ([@banach-space](https://github.com/banach-space))
* ‘arm_sve’ Dialect ([@banach-space](https://github.com/banach-space))
* ‘ArmSME’ Dialect ([@banach-space](https://github.com/banach-space))
* ‘amx’ Dialect ([@adam-smnk](https://github.com/adam-smnk))
* ‘x86vector’ Dialect ([@adam-smnk](https://github.com/adam-smnk))
* ‘vcix’ Dialect ([@mshockwave](https://github.com/mshockwave))

#### Paradigm Dialects
* ‘omp’ Dialect ([@tblah](https://github.com/tblah), [@skatrak](https://github.com/skatrak))
* ‘acc’ Dialect ([@clementval](https://github.com/clementval), [@razvanlupusoru](https://github.com/razvanlupusoru))
* ‘mpi’ Dialect ([@fschlimb](https://github.com/fschlimb))
* ‘shard’ Dialect ([@fschlimb](https://github.com/fschlimb))

## Tensor Compiler

MLIR components specific to construction of compilers for tensor algebra, in
particular for machine learning compilers.

- Renato Golin \
  rengolin@gmail.com (email),
  [@rengolin](https://github.com/rengolin) (GitHub),
  rengolin (Discourse)
- Jacques Pienaar \
  jpienaar@google.com (email),
  [@jpienaar](https://github.com/jpienaar) (GitHub),
  jpienaar (Discourse)
- Andrzej Warzynski \
  andrzej.warzynski@arm.com (email),
  [@banach-space](https://github.com/banach-space) (GitHub),
  banach-space (Discourse)
