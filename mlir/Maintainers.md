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
