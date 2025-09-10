# IRDL Rationale

The IRDL (*Intermediate Representation Definition Language*) dialect allows defining MLIR dialects as MLIR programs. Nested operations are used to represent dialect structure: dialects contain operations, types and attributes, themselves containing type parameters, operands, results, etc. Each of those concepts are mapped to MLIR operations in the IRDL dialect, as shown in the example below:

```mlir
irdl.dialect @my_dialect {
	irdl.type @my_type {
		// This type has a single parameter that must be i32.
		%constraint = irdl.is : i32
		irdl.parameters(param1: %constraint)
	}

	irdl.operation @my_scale_op {
	    // This operation represents the scaling of a vector.
		%vec_constraint = irdl.is : vector<i64>
		%scalar_constraint = irdl.is : i64
		irdl.operands(vector: %vec_constraint, scalar: %scalar_constraint)
		irdl.results(result: %vec_constraint)
	}
}
```

IRDL provides a declarative way to define verifiers using constraint operations (`irdl.is` in the example above). See [constraints and combinators](#constraints-and-combinators) for more details.

The core principles of IRDL are the following, in no particular order:

- **Portability.** IRDL dialects should be self-contained, such that dialects can be easily distributed with minimal assumptions on which compiler infrastructure (or which commit of MLIR) is used.
- **Introspection.** The IRDL dialect definition mechanism should strive towards offering as much introspection abilities as possible. Dialects should be as easy to manipulate, generate, and analyze as possible.
- **Runtime declaration support**. The specification of IRDL dialects should offer the ability to have them be loaded at runtime, via dynamic registration or JIT compilation. Compatibility with dynamic workflows should not hinder the ability to compile IRDL dialects into ahead-of-time declarations.
- **Reliability.** Concepts in IRDL should be consistent and predictable, with as much focus on high-level simplicity as possible. Consequently, IRDL definitions that verify should work out of the box, and those that do not verify should provide clear and understandable errors in all circumstances.

While IRDL simplifies IR definition, it remains an IR itself and thus does not require to be comfortably user-writeable.

## Constraints and combinators

Attribute, type and operation verifiers are expressed in terms of constraint variables. Constraint variables are defined as the results of constraint operations (like `irdl.is` or constraint combinators).

Constraint variables act as variables: as such, matching against the same constraint variable multiple times can only succeed if the matching type or attribute is the same as the one that previously matched. In the following example:

```
irdl.type foo {
	%ty = irdl.any_type
	irdl.parameters(param1: %ty, param2: %ty)
}
```

only types with two equal parameters will successfully match (`foo<i32, i32>` would match while `foo<i32, i64>` would fail, even though both i32 and i64 individually satisfy the `irdl.any_type` constraint). This constraint variable mechanism allows to easily express a requirement on type or attribute equality.

To declare more complex verifiers, IRDL provides constraint-combinator operations such as `irdl.any_of`, `irdl.all_of` or `irdl.parametric`. These combinators can be used to combine constraint variables into new constraint variables. Like all uses of constraint variables, their constraint variable operands enforce equality of matched types of attributes as explained in the previous paragraph.

## Motivating use cases

To illustrate the rationale behind IRDL, the following list describes examples of intended use cases for IRDL, in no particular order:

- **Fuzzer generation.** With declarative verifier definitions, it is possible to compile IRDL dialects into compiler fuzzers that generate only programs passing verifiers.
- **Portable dialects between compiler infrastructures.** Some compiler infrastructures are independent from MLIR but are otherwise IR-compatible. Portable IRDL dialects allow to share the dialect definitions between MLIR and other compiler infrastructures without needing to maintain multiple potentially out-of-sync definitions.
- **Dialect simplification.** Because IRDL definitions can easily be mechanically modified, it is possible to simplify the definition of dialects based on which operations are actually used, leading to smaller compilers.
- **SMT analysis.** Because IRDL dialect definitions are declarative, their definition can be lowered to alternative representations like SMT, allowing analysis of the behavior of transforms taking verifiers into account.
