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

### Code

#### Standalone subcategories
* Core tooling (ODS, DRR, PDLL, LSP) (core)
* CMake ([christopherbate](https://github.com/christopherbate))
* Dialect Conversion ([matthias-springer](https://github.com/matthias-springer), [zero9178](https://github.com/zero9178))
* Python Bindings ([makslevental](https://github.com/makslevental), [rolfmorel](https://github.com/rolfmorel))

### Dialects

#### Code Structure Dialects
* Builtin Dialect (core)
* ‘func’ Dialect (core)
* ‘scf’ Dialect (core)
* ‘cf’ Dialect (core)
* ‘index’ Dialect (core)
* ‘ptr’ Dialect ([fabianmcg](https://github.com/fabianmcg))

#### Basic Compute Dialects
* ‘arith’ Dialect (core)
* ‘math’ Dialect (core)
* Rewrite System Dialects (core)
* Transform Dialect ([martin-luecke](https://github.com/martin-luecke), [ftynse](https://github.com/ftynse), [rolfmorel](https://github.com/rolfmorel))
* ‘pdl_interp’ Dialect ([jpienaar](https://github.com/jpienaar))
* ‘pdl’ Dialect ([jpienaar](https://github.com/jpienaar))

#### Accessory Dialects
* ‘affine’ Dialect ([ftynse](https://github.com/ftynse))
* ‘dlti’ Dialect ([rolfmorel](https://github.com/rolfmorel))
* ‘irdl’ Dialect ([math-fehr](https://github.com/math-fehr), [moxinilian](https://github.com/moxinilian))
* ‘shape’ Dialect ([jpienaar](https://github.com/jpienaar))
* ‘smt’ Dialect ([fabianschuiki](https://github.com/fabianschuiki), [maerhart](https://github.com/maerhart))
* ‘ub’ Dialect ([Hardcode84](https://github.com/Hardcode84))
* ‘complex’ Dialect (core)
* ‘async’ Dialect (unmaintained)

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

### Dialects

The `tensor` maintainer refers to the people working in the tensor compiler category, with the point-of-contact being the maintainers above.
These are key MLIR dialects that will never become _unmaintained_.
Named maintainers, if available, should be contacted first, as they're more active in those areas.

#### Linear Algebra Dialects
* ‘linalg’ Dialect (tensor)
* Tensor Operator Set Architecture (TOSA) Dialect ([@sjarus](https://github.com/sjarus))

#### Type Dialects
* ‘tensor’ Dialect (tensor)
* ‘memref’ Dialect (tensor)
* ‘vector’ Dialect (tensor + [@dcaballe](https://github.com/dcaballe), [@Groverkss](https://github.com/Groverkss))
* ‘sparse_tensor’ Dialect ([@aartbik](https://github.com/aartbik), [@matthias-springer](https://github.com/matthias-springer))

#### Accessory Dialects
* ‘bufferization’ Dialect (tensor, [@matthias-springer](https://github.com/matthias-springer))
* ‘ml_program’ Dialect ([@jpienaar](https://github.com/jpienaar))
* ‘quant’ Dialect (unmaintained)
