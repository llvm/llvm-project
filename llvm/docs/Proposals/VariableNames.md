# Variable Names Plan

This plan is *provisional*. It is not agreed upon. It is written with the
intention of capturing the desires and concerns of the LLVM community, and
forming them into a plan that can be agreed upon.
The original author is somewhat naïve in the ways of LLVM so there will
inevitably be some details that are flawed. You can help - you can edit this
page (preferably with a Phabricator review for larger changes) or reply to the
[Request For Comments thread](http://lists.llvm.org/pipermail/llvm-dev/2019-February/130083.html).

## Too Long; Didn't Read

Improve the readability of LLVM code.

## Introduction

The current [variable naming rule](../CodingStandards.html#name-types-functions-variables-and-enumerators-properly)
states:

> Variable names should be nouns (as they represent state). The name should be
> camel case, and start with an upper case letter (e.g. Leader or Boats).

This rule is the same as that for type names. This is a problem because the
type name cannot be reused for a variable name [^footnote-1]. LLVM developers tend to
work around this by either prepending `The` to the type name:

```
Triple TheTriple;
```

... or more commonly use an acronym, despite the coding standard stating "Avoid
abbreviations unless they are well known":

```
Triple T;
```

The proliferation of acronyms leads to hard-to-read code such as [this](https://github.com/llvm/llvm-project/blob/0a8bc14ad7f3209fe702d18e250194cd90188596/llvm/lib/Transforms/Vectorize/LoopVectorize.cpp#L7445):

```
InnerLoopVectorizer LB(L, PSE, LI, DT, TLI, TTI, AC, ORE, VF.Width, IC,
                       &LVL, &CM);
```

Many other coding guidelines [^cite_lldb] [^cite_google] [^cite_webkit] [^cite_qt] [^cite_rust] [^cite_swift]
[^cite_python] require that variable names begin with a lower case letter in contrast
to class names which begin with a capital letter. This convention means that the
most readable variable name also requires the least thought:

```
Triple triple;
```

There is some agreement that the current rule is broken [^cite_lattneragree]
[^cite_arsenaultagree] [^cite_robinsonagree] and that acronyms are an obstacle to reading
new code [^cite_malyutindistinguish] [^cite_carruthacronym] [^cite_picusacronym]. There are
some opposing views [^cite_parzyszekacronym2] [^cite_ricciacronyms].

This work-in-progress proposal is to change the coding standard for variable
names to require that they start with a lower case letter.

[^footnote-1]: In [some cases](https://github.com/llvm/llvm-project/blob/8b72080d4d7b13072f371712eed333f987b7a18e/llvm/lib/CodeGen/SelectionDAG/SelectionDAG.cpp#L2727)
    the type name *is* reused as a variable name, but this shadows the type name
    and confuses many debuggers [^cite_denisovcamelback].

## Variable Names Coding Standard Options

There are two main options for variable names that begin with a lower case
letter: `camelBack` and `lower_case`. (These are also known by other names
but here we use the terminology from clang-tidy).

`camelBack` is consistent with [^cite_webkit], [^cite_qt] and [^cite_swift] while
`lower_case` is consistent with [^cite_lldb], [^cite_google], [^cite_rust] and [^cite_python].

`camelBack` is already used for function names, which may be considered an
advantage [^cite_lattnerfunction] or a disadvantage [^cite_carruthfunction].

Approval for `camelBack` was expressed by [^cite_denisovcamelback]
[^cite_lattnerfunction] [^cite_ivanovicdistinguish].
Opposition to `camelBack` was expressed by [^cite_carruthcamelback]
[^cite_turnercamelback].
Approval for `lower_case` was expressed by [^cite_carruthlower]
[^cite_carruthcamelback] [^cite_turnerlldb].
Opposition to `lower_case` was expressed by [^cite_lattnerlower].

### Differentiating variable kinds

An additional requested change is to distinguish between different kinds of
variables [^cite_robinsondistinguish] [^cite_robinsondistinguish2] [^cite_jonesdistinguish]
[^cite_ivanovicdistinguish] [^cite_carruthdistinguish] [^cite_malyutindistinguish].

Others oppose this idea [^cite_hahnledistinguish] [^cite_greenedistinguish]
[^cite_hendersonprefix].

A possibility is for member variables to be prefixed with `m_` and for global
variables to be prefixed with `g_` to distinguish them from local variables.
This is consistent with [^cite_lldb]. The `m_` prefix is consistent with [^cite_webkit].

A variation is for member variables to be prefixed with `m`
[^cite_ivanovicdistinguish] [^cite_beylsdistinguish]. This is consistent with [^cite_mozilla].

Another option is for member variables to be suffixed with `_` which is
consistent with [^cite_google] and similar to [^cite_python]. Opposed by
[^cite_parzyszekdistinguish].

## Reducing the number of acronyms

While switching coding standard will make it easier to use non-acronym names for
new code, it doesn't improve the existing large body of code that uses acronyms
extensively to the detriment of its readability. Further, it is natural and
generally encouraged that new code be written in the style of the surrounding
code. Therefore it is likely that much newly written code will also use
acronyms despite what the coding standard says, much as it is today.

As well as changing the case of variable names, they could also be expanded to
their non-acronym form e.g. `Triple T` → `Triple triple`.

There is support for expanding many acronyms [^cite_carruthacronym] [^cite_picusacronym]
but there is a preference that expanding acronyms be deferred
[^cite_parzyszekacronym] [^cite_carruthacronym].

The consensus within the community seems to be that at least some acronyms are
valuable [^cite_parzyszekacronym] [^cite_lattneracronym]. The most commonly cited acronym
is `TLI` however that is used to refer to both `TargetLowering` and
`TargetLibraryInfo` [^cite_greenedistinguish].

The following is a list of acronyms considered sufficiently useful that the
benefit of using them outweighs the cost of learning them. Acronyms that are
either not on the list or are used to refer to a different type should be
expanded.

| Class name                   | Variable name |
| ---------------------------- | ------------- |
| DeterministicFiniteAutomaton | dfa           |
| DominatorTree                | dt            |
| LoopInfo                     | li            |
| MachineFunction              | mf            |
| MachineInstr                 | mi            |
| MachineRegisterInfo          | mri           |
| ScalarEvolution              | se            |
| TargetInstrInfo              | tii           |
| TargetLibraryInfo            | tli           |
| TargetRegisterInfo           | tri           |

In some cases renaming acronyms to the full type name will result in overly
verbose code. Unlike most classes, a variable's scope is limited and therefore
some of its purpose can implied from that scope, meaning that fewer words are
necessary to give it a clear name. For example, in an optimization pass the reader
can assume that a variable's purpose relates to optimization and therefore an
`OptimizationRemarkEmitter` variable could be given the name `remarkEmitter`
or even `remarker`.

The following is a list of longer class names and the associated shorter
variable name.

| Class name                | Variable name |
| ------------------------- | ------------- |
| BasicBlock                | block         |
| ConstantExpr              | expr          |
| ExecutionEngine           | engine        |
| MachineOperand            | operand       |
| OptimizationRemarkEmitter | remarker      |
| PreservedAnalyses         | analyses      |
| PreservedAnalysesChecker  | checker       |
| TargetLowering            | lowering      |
| TargetMachine             | machine       |

## Transition Options

There are three main options for transitioning:

1. Keep the current coding standard
2. Laissez faire
3. Big bang

### Keep the current coding standard

Proponents of keeping the current coding standard (i.e. not transitioning at
all) question whether the cost of transition outweighs the benefit
[^cite_emersonconcern] [^cite_reamesconcern] [^cite_bradburyconcern].
The costs are that `git blame` will become less usable; and that merging the
changes will be costly for downstream maintainers. See [Big bang] for potential
mitigations.

### Laissez faire

The coding standard could allow both `CamelCase` and `camelBack` styles for
variable names [^cite_lattnertransition].

A code review to implement this is at <https://reviews.llvm.org/D57896>.

#### Advantages

> - Very easy to implement initially.

#### Disadvantages

> - Leads to inconsistency [^cite_bradburyconcern] [^cite_aminiinconsistent].
> - Inconsistency means it will be hard to know at a guess what name a variable
>   will have [^cite_dasinconsistent] [^cite_carruthinconsistent].
> - Some large-scale renaming may happen anyway, leading to its disadvantages
>   without any mitigations.

### Big bang

With this approach, variables will be renamed by an automated script in a series
of large commits.

The principle advantage of this approach is that it minimises the cost of
inconsistency [^cite_bradburytransition] [^cite_robinsontransition].

It goes against a policy of avoiding large-scale reformatting of existing code
[^cite_greenedistinguish].

It has been suggested that LLD would be a good starter project for the renaming
[^cite_ueyama].

#### Keeping git blame usable

`git blame` (or `git annotate`) permits quickly identifying the commit that
changed a given line in a file. After renaming variables, many lines will show
as being changed by that one commit, requiring a further invocation of `git
blame` to identify prior, more interesting commits [^cite_greenegitblame]
[^cite_ricciacronyms].

**Mitigation**: [git-hyper-blame](https://commondatastorage.googleapis.com/chrome-infra-docs/flat/depot_tools/docs/html/git-hyper-blame.html)
can ignore or "look through" a given set of commits.
A `.git-blame-ignore-revs` file identifying the variable renaming commits
could be added to the LLVM git repository root directory.
It is being [investigated](https://public-inbox.org/git/20190324235020.49706-1-michael@platin.gs/)
whether similar functionality could be added to `git blame` itself.

#### Minimising cost of downstream merges

There are many forks of LLVM with downstream changes. Merging a large-scale
renaming change could be difficult for the fork maintainers.

**Mitigation**: A large-scale renaming would be automated. A fork maintainer can
merge from the commit immediately before the renaming, then apply the renaming
script to their own branch. They can then merge again from the renaming commit,
resolving all conflicts by choosing their own version. This could be tested on
the [^cite_sve] fork.

## Provisional Plan

This is a provisional plan for the [Big bang] approach. It has not been agreed.

01. Investigate improving `git blame`. The extent to which it can be made to
    "look through" commits may impact how big a change can be made.

02. Write a script to expand acronyms.

03. Experiment and perform dry runs of the various refactoring options.
    Results can be published in forks of the LLVM Git repository.

04. Consider the evidence and agree on the new policy.

05. Agree & announce a date for the renaming of the starter project (LLD).

06. Update the [policy page](../CodingStandards.html). This will explain the
    old and new rules and which projects each applies to.

07. Refactor the starter project in two commits:

    1. Add or change the project's .clang-tidy to reflect the agreed rules.
       (This is in a separate commit to enable the merging process described in
       [Minimising cost of downstream merges]).
       Also update the project list on the policy page.

    2. Apply `clang-tidy` to the project's files, with only the
       `readability-identifier-naming` rules enabled. `clang-tidy` will also
       reformat the affected lines according to the rules in `.clang-format`.
       It is anticipated that this will be a good dog-fooding opportunity for
       clang-tidy, and bugs should be fixed in the process, likely including:

       > - [readability-identifier-naming incorrectly fixes lambda capture](https://bugs.llvm.org/show_bug.cgi?id=41119).
       > - [readability-identifier-naming incorrectly fixes variables which
       >   become keywords](https://bugs.llvm.org/show_bug.cgi?id=41120).
       > - [readability-identifier-naming misses fixing member variables in
       >   destructor](https://bugs.llvm.org/show_bug.cgi?id=41122).

08. Gather feedback and refine the process as appropriate.

09. Apply the process to the following projects, with a suitable delay between
    each (at least 4 weeks after the first change, at least 2 weeks subsequently)
    to allow gathering further feedback.
    This list should exclude projects that must adhere to an externally defined
    standard e.g. libcxx.
    The list is roughly in chronological order of renaming.
    Some items may not make sense to rename individually - it is expected that
    this list will change following experimentation:

    - TableGen
    - llvm/tools
    - clang-tools-extra
    - clang
    - ARM backend
    - AArch64 backend
    - AMDGPU backend
    - ARC backend
    - AVR backend
    - BPF backend
    - Hexagon backend
    - Lanai backend
    - MIPS backend
    - NVPTX backend
    - PowerPC backend
    - RISC-V backend
    - Sparc backend
    - SystemZ backend
    - WebAssembly backend
    - X86 backend
    - XCore backend
    - libLTO
    - Debug Information
    - Remainder of llvm
    - compiler-rt
    - libunwind
    - openmp
    - parallel-libs
    - polly
    - lldb

10. Remove the old variable name rule from the policy page.

11. Repeat many of the steps in the sequence, using a script to expand acronyms.

## References

[^cite_lldb]: LLDB Coding Conventions <https://llvm.org/svn/llvm-project/lldb/branches/release_39/www/lldb-coding-conventions.html>

[^cite_google]: Google C++ Style Guide <https://google.github.io/styleguide/cppguide.html#Variable_Names>

[^cite_webkit]: WebKit Code Style Guidelines <https://webkit.org/code-style-guidelines/#names>

[^cite_qt]: Qt Coding Style <https://wiki.qt.io/Qt_Coding_Style#Declaring_variables>

[^cite_rust]: Rust naming conventions <https://doc.rust-lang.org/1.0.0/style/style/naming/README.html>

[^cite_swift]: Swift API Design Guidelines <https://swift.org/documentation/api-design-guidelines/#general-conventions>

[^cite_python]: Style Guide for Python Code <https://www.python.org/dev/peps/pep-0008/#function-and-variable-names>

[^cite_mozilla]: Mozilla Coding style: Prefixes <https://firefox-source-docs.mozilla.org/code-quality/coding-style/coding_style_cpp.html#prefixes>

[^cite_sve]: LLVM with support for SVE <https://github.com/ARM-software/LLVM-SVE>

[^cite_aminiinconsistent]: Mehdi Amini, <http://lists.llvm.org/pipermail/llvm-dev/2019-February/130329.html>

[^cite_arsenaultagree]: Matt Arsenault, <http://lists.llvm.org/pipermail/llvm-dev/2019-February/129934.html>

[^cite_beylsdistinguish]: Kristof Beyls, <http://lists.llvm.org/pipermail/llvm-dev/2019-February/130292.html>

[^cite_bradburyconcern]: Alex Bradbury, <http://lists.llvm.org/pipermail/llvm-dev/2019-February/130266.html>

[^cite_bradburytransition]: Alex Bradbury, <http://lists.llvm.org/pipermail/llvm-dev/2019-February/130388.html>

[^cite_carruthacronym]: Chandler Carruth, <http://lists.llvm.org/pipermail/llvm-dev/2019-February/130313.html>

[^cite_carruthcamelback]: Chandler Carruth, <http://lists.llvm.org/pipermail/llvm-dev/2019-February/130214.html>

[^cite_carruthdistinguish]: Chandler Carruth, <http://lists.llvm.org/pipermail/llvm-dev/2019-February/130310.html>

[^cite_carruthfunction]: Chandler Carruth, <http://lists.llvm.org/pipermail/llvm-dev/2019-February/130309.html>

[^cite_carruthinconsistent]: Chandler Carruth, <http://lists.llvm.org/pipermail/llvm-dev/2019-February/130312.html>

[^cite_carruthlower]: Chandler Carruth, <http://lists.llvm.org/pipermail/llvm-dev/2019-February/130430.html>

[^cite_dasinconsistent]: Sanjoy Das, <http://lists.llvm.org/pipermail/llvm-dev/2019-February/130304.html>

[^cite_denisovcamelback]: Alex Denisov, <http://lists.llvm.org/pipermail/llvm-dev/2019-February/130179.html>

[^cite_emersonconcern]: Amara Emerson, <http://lists.llvm.org/pipermail/llvm-dev/2019-February/129894.html>

[^cite_greenedistinguish]: David Greene, <http://lists.llvm.org/pipermail/llvm-dev/2019-February/130425.html>

[^cite_greenegitblame]: David Greene, <http://lists.llvm.org/pipermail/llvm-dev/2019-February/130228.html>

[^cite_hendersonprefix]: James Henderson, <http://lists.llvm.org/pipermail/llvm-dev/2019-February/130465.html>

[^cite_hähnledistinguish]: Nicolai Hähnle, <http://lists.llvm.org/pipermail/llvm-dev/2019-February/129923.html>

[^cite_ivanovicdistinguish]: Nemanja Ivanovic, <http://lists.llvm.org/pipermail/llvm-dev/2019-February/130249.html>

[^cite_jonesdistinguish]: JD Jones, <http://lists.llvm.org/pipermail/llvm-dev/2019-February/129926.html>

[^cite_lattneracronym]: Chris Lattner, <http://lists.llvm.org/pipermail/llvm-dev/2019-February/130353.html>

[^cite_lattneragree]: Chris Latter, <http://lists.llvm.org/pipermail/llvm-dev/2019-February/129907.html>

[^cite_lattnerfunction]: Chris Lattner, <http://lists.llvm.org/pipermail/llvm-dev/2019-February/130630.html>

[^cite_lattnerlower]: Chris Lattner, <http://lists.llvm.org/pipermail/llvm-dev/2019-February/130629.html>

[^cite_lattnertransition]: Chris Lattner, <http://lists.llvm.org/pipermail/llvm-dev/2019-February/130355.html>

[^cite_malyutindistinguish]: Danila Malyutin, <http://lists.llvm.org/pipermail/llvm-dev/2019-February/130320.html>

[^cite_parzyszekacronym]: Krzysztof Parzyszek, <http://lists.llvm.org/pipermail/llvm-dev/2019-February/130306.html>

[^cite_parzyszekacronym2]: Krzysztof Parzyszek, <http://lists.llvm.org/pipermail/llvm-dev/2019-February/130323.html>

[^cite_parzyszekdistinguish]: Krzysztof Parzyszek, <http://lists.llvm.org/pipermail/llvm-dev/2019-February/129941.html>

[^cite_picusacronym]: Diana Picus, <http://lists.llvm.org/pipermail/llvm-dev/2019-February/130318.html>

[^cite_reamesconcern]: Philip Reames, <http://lists.llvm.org/pipermail/llvm-dev/2019-February/130181.html>

[^cite_ricciacronyms]: Bruno Ricci, <http://lists.llvm.org/pipermail/llvm-dev/2019-February/130328.html>

[^cite_robinsonagree]: Paul Robinson, <http://lists.llvm.org/pipermail/llvm-dev/2019-February/130111.html>

[^cite_robinsondistinguish]: Paul Robinson, <http://lists.llvm.org/pipermail/llvm-dev/2019-February/129920.html>

[^cite_robinsondistinguish2]: Paul Robinson, <http://lists.llvm.org/pipermail/llvm-dev/2019-February/130229.html>

[^cite_robinsontransition]: Paul Robinson, <http://lists.llvm.org/pipermail/llvm-dev/2019-February/130415.html>

[^cite_turnercamelback]: Zachary Turner, <https://reviews.llvm.org/D57896#1402264>

[^cite_turnerlldb]: Zachary Turner, <http://lists.llvm.org/pipermail/llvm-dev/2019-February/130213.html>

[^cite_ueyama]: Rui Ueyama, <http://lists.llvm.org/pipermail/llvm-dev/2019-February/130435.html>

