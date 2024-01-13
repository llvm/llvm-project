
## MPACT Microarchitecture Description Language

Reid Tatge          [tatge@google.com](mailto:tatge@google.com)


[TOC]



### **Goals for a Machine Description Language**

Modern processors are complex: multiple execution pipelines, dynamically dispatched, out-of-order execution, register renaming, forwarding networks, and (often) undocumented micro-operations. Instruction behaviors, including micro-operations, often can’t be _statically_ modeled in an accurate way, but only _statistically_ modeled. In these cases, the compiler’s model of a microarchitecture (Schedules and Itineraries in LLVM) is effectively closer to a heuristic than a formal model. And this works quite well for general purpose microprocessors.

However, modern accelerators have different and/or additional dimensions of complexity: VLIW instruction issue, unprotected pipelines, tensor/vector ALUs, software-managed memory hierarchies. And it's more critical that compilers can precisely model the details of that complexity. Currently, LLVM’s Schedules and Itineraries aren’t adequate for directly modeling many accelerator architectural features.

So we have several goals:



1. We want a first-class, purpose-built, intuitive language that captures all the scheduling and latency details of the architecture - much like Schedules and Itineraries - that works well for all current targets, but also for a large class of accelerator architectures..
2. The complexity of the specification should scale with the complexity of the hardware. 
3. The description should be succinct, avoiding duplicated information, while reflecting the way things are defined in a hardware architecture specification.
4. We want to generate artifacts that can be used in a machine-independent way for back-end optimization, register allocation, instruction scheduling, etc - anything that depends on the behavior and constraints of instructions.
5. We want to support a much larger class of architectures in one uniform manner.

For this document (and language), the term “instructions” refers to the documented instruction set of the machine, as represented by LLVM instructions descriptions, rather than undocumented micro-operations used by many modern microprocessors. 

 

The process of compiling a processor’s machine description creates several primary artifacts:



*   For each target instruction (described in td files), we create an object that describes the detailed behaviors of the instruction in any legal context (for example, on any functional unit, on any processor)
*   A set of methods with machine independent APIs that leverage the information associated with instructions to inform and guide back-end optimization passes.  

The details of the artifacts are described later in this document.

_Note: A full language grammar description is provided in an appendix.  Snippets of grammar throughout the document only provide the pertinent section of the grammar, see the Appendix A for the full grammar._

The proposed language can be thought of as an _optional extension to the LLVM machine description_. For most upstream architectures, the new language offers minimal benefit other than a much more succinct way to specify the architecture vs Schedules and Itineraries.  But for accelerator-class architectures, it provides a level of detail and capability not available in the existing tablegen approaches.


#### **Background**

Processor families evolve over time. They accrete new instructions, and pipelines change - often in subtle ways - as they accumulate more functional units and registers; encoding rules change; issue rules change. Understanding, encoding, and using all of this information - over time, for many subtargets - can be daunting.  When the description language isn’t sufficient to model the architecture, the back-end modeling evolves towards heuristics, and leads to performance issues or bugs in the compiler. And it certainly ends with large amounts of target specific code to handle “special cases”. 

LLVM uses the [TableGen](https://llvm.org/docs/TableGen/index.html) language to describe a processor, and this is quite sufficient for handling most general purpose architectures - there are 20+ processor families currently upstreamed in LLVM! In fact, it is very good at modeling instruction definitions, register classes, and calling conventions.  However, there are “features” of modern accelerator micro-architectures which are difficult or impossible to model in tablegen.

We would like to easily handle:



*   Complex pipeline behaviors
    *   An instruction may have different latencies, resource usage, and/or register constraints on different functional units or different operand values.
    *   An instruction may read source registers more than once (in different pipeline phases).
    *   Pipeline structure, depth, hazards, scoreboarding, and protection may differ between family members.
*   Functional units
    *   Managing functional unit behavior differences across subtargets of a family.
    *   Impose different register constraints on instructions (local register files, for example).
    *   Share execution resources with other functional units (such as register ports)
    *   Functional unit clusters with separate execution pipelines.
*   VLIW Architecture 
    *   issue rules can get extremely complex, and can be dependent on encoding, operand features, and pipeline behavior of candidate instructions.** \
**

More generally, we’d like specific language to:



*   Support all members of a processor family
*   Describe CPU features, parameterized by subtarget
    *   Functional units
    *   Issue slots
    *   Pipeline structure and behaviors

Since our emphasis is on easily supporting accelerators and VLIW processors, in addition to supporting all existing targets, much of this is overkill for most upstreamed CPUs.  CPU’s typically have much simpler descriptions, and don’t require much of the capability of our machine description language.  Incidentally, MDL descriptions of these targets (generated automatically from the tablegen Schedules and Itineraries) are typically much more concise than the original tablegen descriptions.


#### **Approach - “Subunits” and Instruction Behaviors**

We developed a DSL that allows us to describe an arbitrary processor microarchitecture in terms that reflect what is typically documented in the hardware specification. The MDL compiler creates a database that provides microarchitecture behavior information that can _automatically _inform critical back-end compiler passes, such as instruction scheduling and register allocation, in a machine-independent way. 

It’s important to note the difference between an instruction definition, as described in LLVM, and an instruction instance.  Generally, instructions defined in LLVM share the same behaviors across all instances of that instruction in a single subtarget. Exceptions to this require non-trivial code in the back-end to model variant behavior.  In VLIW and accelerator architectures, each generated instance of an instruction can have different behaviors, depending on how it's issued, its operand values, the functional unit it runs on, and the subtarget. So we provide a way to model those differences in reasonable ways.

The MDL introduces the concept of a “subunit” to abstractly represent a class of instructions with the same behaviors. Subunit instances concretely connect instructions to descriptions of their behaviors, _and_ to the functional units that they can be issued on. A subunit is vaguely analogous to collections of SchedRead and SchedWrite resources. 

Naively, we could create unique subunits for each behavior for each instruction, the set of which would enumerate the cross-product of the instruction’s behaviors on every subtarget, functional unit, and issue slot. But subunits can be specialized by subtarget, functional unit, and each instruction definition, so a single subunit definition can properly describe behaviors for sets of instructions in many different contexts.

A key aspect of this language design is that we can explicitly represent the potentially polymorphic behavior of each generated instance of any instruction, on any functional unit, on any subtarget.  The representation also comprehends that this information can vary between each of an instruction’s instances.

  

We define a subunit as an object that defines the _behavior sets_ of an instruction instance in all legal contexts (functional units, issue slots), for each subtarget.  In particular, we want to know: \




*   What resources are shared or reserved, in what pipeline phases.
    *   Encoding resources
    *   Issue slot(s) used
    *   Functional unit resources
    *   Shared/private busses, register ports, resources, or pooled resources
*   What registers are read and written, in which pipeline phases (ie, the instruction’s “latencies”)
*   What additional register constraints does a functional unit instance impose on an instruction’s registers.

The critical artifact generated by the MDL compiler is a set of instruction behaviors for each instruction definition.  For each subtarget, for each instruction, we generate a list of every possible behavior of that instruction on that CPU.  While this sounds daunting, in practice it's rare to have more than a few behaviors for an instruction, and most instruction definitions share their behaviors with many other instructions, across subtargets.


### **Overview of a Processor Family Description**

This document generally describes the language in a bottom up order - details first.  But let's start with a brief tops-down overview of what a processor family description looks like, without going into details about each part.

A minimal processor family description has the following components:



*   A set of CPU definitions - one for each subtarget.
*   A set of functional unit template definitions,
*   A set of subunit template definitions,
*   A set of latency template definitions.

A CPU definition specifies a set of functional unit instances that define the processor, as well as pipeline descriptions, issue slot resources, and binding of functional units to issue slots.  Each functional unit instance can be parameterized and specialized.

A functional unit template specifies a set of subunits instances implemented by an instance of the functional unit.  It can be parameterized and specialized for each instance in different CPUs.

A subunit template abstractly defines a set of related operations that have similar behaviors. They specify these behaviors with a set of “latency” instances.  They can also be parameterized and specialized for each instance in different functional unit templates.  Subunits tie instruction definitions both to functional units on which they can execute, and instruction behaviors described in latency templates.

A latency template defines the pipeline behavior of a set of instructions.  It can be parameterized and specialized for each instance in a subunit instance.  It is also specialized for each instruction that is tied to it (through a subunit).  A latency rule, at a minimum, specifies when each operand is read and written in the execution pipeline.

Here’s a very simple example of a trivial CPU, with three functional units, two issue slots, and a four-deep pipeline:


```
    cpu myCpu {
    	phases cpu { E1, E2, E3, E4 };
    issue slot1, slot2;
    	func_unit FU_ALU my_alu1();    	// an instance of FU_ALU
    	func_unit FU_ALU my_alu2();    	// an instance of FU_ALU
    	func_unit FU_LOAD my_load();   	// an instance of FU_LOAD
    }

    func_unit FU_ALU() {            		// template definition for FU_ALU
    	subunit ALU();              	// an instance of subunit ALU
    }
    func_unit FU_LOAD() {               	// template definition for FU_LOAD
    	subunit LOAD();                	// an instance of subunit LOAD
    }

    subunit ALU() {                      	// template definition for ALU
    	latency LALU();                	// an instance of latency LALU
    }
    subunit LOAD() {                     	// template definition for LOAD
    	latency LLOAD();               	// an instance of latency LLOAD
    }

    latency LALU() {                     	// template definition for LALU
    	def(E2, $dst);  use(E1, $src1);  use(E1, $src2); 
    }
    latency LLOAD() {                    	// template definition for LLOAD
    	def(E4, $dst);  use(E1, $addr);
    }
```


A more complete description of each part of this description is provided in the section “Defining a Processor Family”.

**Defining an ISA**

We need to map a microarchitecture model back to LLVM instruction, operand, and register definitions.  So, the MDL contains constructs for defining instructions, operands, registers, and register classes.  

When writing a target machine description, its not necessary to write descriptions for instructions, operands, and registers - we scrape all of this information about the CPU ISA from the tablegen output as part of the build process, and produce an MDL file which contains these definitions. The machine description compiler uses these definitions to tie architectural information back to LLVM instructions, operands, and register classes.

We will describe these language features here, primarily for completeness.


#### **Defining Instructions**

Instruction definitions are scraped from tablegen files, and provide the following information to the MDL compiler for each instruction:



*   The instruction’s name (as defined in the td files)
*   Its operands, with the operand type and name provided in the order they are declared, and indicating whether each is an input or output of the instruction.
*   A set of “legal” subunit definitions (a “subunit” is described later in this document)
*   An optional list of instructions derived from this one.

As in tablegen, an operand type must be either an operand name defined in the td description, a register class name defined in the td description, or simply a defined register name. If the operand type is a register name, the operand name is optional (and ignored) (these register operands are used to represent implied operands in LLVM instructions). 

Grammar:


```
    instruction_def  : 'instruction' IDENT
                          '(' (operand_decl (',' operand_decl)*)? ')'
                          '{'
                              ('subunit' '(' name_list ')' ';' )?
                              ('derived' '(' name_list ')' ';' )?
                          '}' ';'? ;
    operand_decl     : ((IDENT (IDENT)?) | '...') ('(I)' | '(O)')? ;
```


An example:


```
    instruction ADDSWri(GPR32 Rd(O), GPR32sp Rn(I), addsub_shifted_imm32 imm(I), NZCV(O)) {
      subunit(sub24,sub26);
    }
```


This describes an ARM add instruction that has two defined input operands (Rn, imm), one defined output operand (Rd), and one implicit output operand (NZCV), which is associated with two subunits (sub24, sub26).


#### **Defining Operands**

Operand definitions are scraped from tablegen files (like instructions), and provide the following information to the MDL compiler for each operand:



*   The operand’s name,
*   Its sub-operands, with the operand type and operand name provided in the order they are declared.  Note that operand names are optional, and if not present we would refer to these by their sub-operand id (0, 1, etc),
*   The operand’s value type.

As in LLVM, an operand definition’s sub-operand types may in turn refer to other operand definitions. (Note that operand’s sub-operands are declared with the same syntax as instruction operands.)

Grammar:


```
    operand_def      : 'operand' IDENT
                          '(' (operand_decl (',' operand_decl)*)? ')'
                          '{' operand_type '}' ';'? ;
```


Some examples:


```
    operand GPR32z(GPR32 reg) { type(i32); } 
    operand addsub_shifted_imm32(i32imm, i32imm) { type(i32); }
```



#### **Defining Registers and Register Classes**

Registers and register classes are scraped from tablegen output.  We provide a general method in the language to define registers and classes of registers which can reflect the registers defined in tablegen. 

Grammar:


```
    register_def     : 'register' register_decl (',' register_decl)* ';' ;
    register_decl    : IDENT ('[' range ']')? ;
    register_class   : 'register_class' IDENT
                            '{' register_decl (',' register_decl)* '}' ';'? 
                     | 'register_class' IDENT '{' '}' ';'? ;
```


Examples:


```
    register a0, a1, a2, a3;                 // 4 registers
    register a[4..7];                        // definition of a4, a5, a6, and a7

    register_class low3 { a0, a1, a2 };      // a class of 3 registers
    register_class high5 { a[3..7] };        // a class of a3, a4, a5, a6, and a7
```


The order of register definitions is generally insignificant in the current MDL - we use the register names defined in LLVM, and there’s no cases in the MDL where we depend on order.  Register “ranges”, such as “a[0..20]” are simply expanded into the discrete names of the entire range of registers.


#### **Defining Derived Operands**

LLVM doesn’t necessarily provide all the information we want to capture about an instruction, so the MDL allows for defining “derived” operands with which we can associate named values.  A derived operand is essentially an alias to one or more LLVM-defined operands (or derived operands), and provides a mechanism to add arbitrary attributes to operand definitions. Derived operands also allow us to treat a set of operand types as identical in latency reference rules (so you don’t have to specify a long set of operand types for some references.)

Grammar:


```
    derived_operand_def     : 'operand' IDENT (':' IDENT)+  ('(' ')')?
                                  '{' (operand_type | operand_attribute)* '}' ';'? ;
    operand_attribute_stmt  : 'attribute' IDENT '=' (snumber | tuple)
                                ('if' ('lit' | 'address' | 'label')
```



    `                              ('[' pred_value (',' pred_value)* ']' )? )? ';' `;


```
    pred_value              : snumber
                            | snumber '..' snumber
                            | '{' number '}' ;
	tuple                   : '[' snumber (',' snumber)* ']' ;
```



##### **Derivation**

Each derived operand is declared with one or more “base” operands, for which it is an alias. Circular or ambiguous derivations are explicitly disallowed - there must be only one derivation path for a derived operand to any of its base concrete operands.

Derived operands are used in place of their base operands in operand latency rules in latency templates (described later). This allows a rule to match a set of operands, rather than a single operand, and also can provide access to instruction attributes to the latency rule.


##### **Derived operand attributes**

Derived operand attributes associate name/value-tuple pairs with the operand type. Tuples are appropriate when an attribute is used as a set of masks for resource sharing, described later.  

Some examples:


```
    attribute my_attr_a = 1;
    attribute my_attr_b = 123;
    attribute my_tuple  = [1, 2, 3];
```


Attributes can have predicates that check if the operand contains a data address, a code address, or any constant.  Additionally, attributes can have multiple definitions with different predicates, with the first “true” predicate determining the final value of the attribute for that operand instance:


```
    attribute my_attr = 5 if address;    // if operand is a relocatable address
    attribute my_attr = 2 if label;      // if operand is a code address
    attribute my_attr = 3 if lit;        // if operand is any literal constant
```


Predicates for literal constants can also take an optional list of “predicate values”, where each predicate value is either an integer, a range of integers, or a “mask”. Mask predicate values are explicitly checking for non-zero bits:


```
    attribute my_attr = 5 if lit [1, 2, 4, 8];    // looking for specific values
    attribute my_attr = 12 if lit [100..200];     // looking for a range of values
    attribute my_attr = 1 if lit [{0x0000FFFF}];  // looking for a 16 bit number
    attribute my_attr = 2 if lit [{0x00FFFF00}];  // also a 16-bit number!
    attribute my_attr = 3 if lit [1, 4, 10..14, 0x3F800000, {0xFF00FF00}]; 
```


Note that we explicitly don’t directly support floating point numbers: this should be done instead with specific bit patterns or masks.  This avoids problems with floating point precision and format differences across systems:


```
    attribute my_attr = 1 if lit [0xBF800000, 0x402DF854];   // -1.0, or pi
    attribute my_attr = 2 if lit [{0x7FFF000}];              // +BF16 number
```


If all of an attribute’s predicates are “false” for an instance of an operand, the compiler recursively checks the attribute’s value in each of the operand’s bases until if finds a true predicate (or an unpredicated attribute): 


```
    operand i32imm() { type(i32); }   // scraped from llvm td file.

    operand huge_imm : i32imm() {
       attribute size = 3;
    }
    operand medium_imm : big_imm() {
       attribute size = 2 if lit [-32768..32676];
    }
    operand small_imm : medium_imm() {
       attribute size = 1 if lit [0-16];
    }
```



##### **Derived operand attribute usage**

There is currently only a single context in which instruction attributes are used directly in the machine description, as part of resource references in latency rules (see “latency\_resource\_ref”). In this context, you can specify an attribute name which provides the number of resources needed for a resource allocation, and the mask used to determine shared operand bits associated with the resource.  An example:


```
    … my_resource:my_size_attribute:my_mask_attribute …
```


This resource reference uses the attributes from the operand associated with this reference to determine how many resources to allocate, and what bits in the operand to share.


### **Overview of Resources**

Resources are used to abstractly describe hardware constructs that are used by an instruction in its execution.  They can represent:



*   functional units, 
*   issue slots, 
*   register ports, 
*   shared encoding bits, 
*   or can name any hardware resource an instruction uses when it executes that could impact the instruction’s behavior (such as pipeline hazards).

Its important to note that different instances of an instruction can use completely different resources depending on which functional unit, and which subtarget, it's issued on. The MDL has an explicit way to model this.

The machine description provides a mechanism for defining and associating resources with the pipeline behaviors of instructions through the specialization of functional unit templates, subunit templates, and latency templates. It also allows automatic allocation of shared resources for an instruction instance from resource pools. The MDL compiler generates behavior descriptions which explicitly reference each resource (or resource pool) the instruction uses, and in what pipeline phases.  This provides a direct methodology for managing instruction issue and pipeline behaviors such as hazards.


#### Defining Resources

There are a few ways that resources are defined:



*   **Functional Units:** A resource is implicitly defined for every functional unit instance in a CPU definition. An instruction that executes on a particular instance will reserve that resource implicitly. 
*   **Issue Slots: **Each CPU, or cluster of functional units in a CPU, can explicitly define a set of issue slots.  For a VLIW, these resources directly correspond to instruction encoding slots in the machine instruction word, and can be used to control which instruction slots can issue to which functional units.  For dynamically scheduled CPUs, these correspond to the width of the dynamic instruction issue. 
*   **Named Resources** can be explicitly defined in several contexts, described below.
*   **Ports:** Ports are functional unit resources that model a register class constraint and a set of associated resources. These are intended to model register file ports that are shared between functional units.

Explicitly defined resources have scope - they can be defined globally (and apply to all CPU variants), within a CPU, within a cluster, or within a functional unit template.  Intuitively, shared resources are typically defined at higher levels in the machine description hierarchy.  Resources and ports defined within a functional unit template are replicated for each instance of that functional unit.  “Issue” resources are defined in CPU and cluster instances.

Named resource definitions have the following grammar:


```
    resource_def            : 'resource' ('(' IDENT ')')?
                                  resource_decl (',' resource_decl)*  ';' ;
    resource_decl           : IDENT (':' number)? ('[' number ']')?
                            | IDENT (':' number)? '{' name_list '}'
                            | IDENT (':' number)? '{' group_list '}' ;

    port_def                : 'port' port_decl (',' port_decl)* ';' ;
    port_decl               : IDENT ('<' IDENT '>')? ('(' resource_refs ')')? ;
    issue_resource          : 'issue' ('(' IDENT ')')? name_list ';' ;
```



##### Simple resource definitions

The simplest resource definition is simply a comma-separated list of names:

	**<code>resource name1, name2, name3;</code></strong>

A resource can also have an explicit pipeline stage associated with it, indicating that the defined resources are always used in the specified pipeline phase:

	**<code>resource(E4) name1, name2;    // define resources that are always used in E4</code></strong>

A resource can have a set of bits associated with it. This defines a resource that can be shared between two references if the bits in an associated operand reference are identical.

	**<code>resource immediate:8;         // define a resource with 8 bits of data</code></strong>


##### Grouped resource definitions

We can declare a set of named, related resources:

	**<code>resource bits     { bits_1, bits_2, bits_3 };</code></strong>

A resource group typically represents a pool of resources that are shared between instructions executing in parallel, where an instruction may require one or all of the resources. This is a common attribute of VLIW architectures, and used to model things like immediate pools and register ports.

Any defined resource can be included in a group, and the order of the members of a group is significant when members are allocated.  If a group mentions an undefined resource (in either the current or enclosing scope), the member is declared as a resource in the current scope.  In the case above, if the members (bits\_1, etc) are not declared, the compiler would create the definition:

	**<code>resource bits_1, bits_2, bits_3;</code></strong>

and the group members would refer to these definitions. (Note: we don’t support nested groups).

The resource group can be referenced by name, referring to the entire pool, or by individual members, such as “bits.bits\_2” to specify the use of a specific pooled resource.  Consider the following example:


```
	resource bits_1, bits_2, bits_3;
```


	**<code>resource bits_x { bits_1, bits_2, bits_3 };</code></strong>

	**<code>resource bits_y { bits_3, bits_1, bits_2 };</code></strong>

“bits\_x” and “bits\_y” are distinct groups that reference the same members, but members are allocated in a different order.  Groups can also be defined with syntax that indicates how its members are allocated by default.

	**<code>resource bits_or  { bits_1 | bits_2 | bits_3 };       // allocate one of these</code></strong>

	**<code>resource bits_and { bits_1 & bits_2 & bits_3 };       // allocate all of these</code></strong>

Groups can also be implicitly defined in functional unit and subunit template instantiations as a resource parameter.

	**<code>func_unit func my_fu(bits_1 | bits_2 | bits_3);</code></strong>

This implicitly defines a resource group with three members, and passes that group as a parameter of the instance.


##### Pooled resource definitions

We can also declare a set of “unnamed” pooled resources:


```
	resource shared_bits[0..5];
```


This describes a resource pool with 6 members.  The entire pool can be referenced by name (ie “shared\_bits”), or each member can be referenced by index (“shared\_bits[3]”), or a subrange of members (“shared\_bits[2..3]). A resource reference can also indicate that it needs some number of resources allocated with the syntax: shared\_bits:<number>.  

Resource pools can also have data associated with them, each member has its own set of bits:


```
resource bits:20 { bits_1, bits_2, bits_3 };
resource shared_bits:5[6];
```


Resource pools, like resource groups, are used to model things like shared encoding bits and shared register ports, where instructions need one or more members of a set of pooled resources.

Finally, resource definitions can pin a resource to a particular pipeline phase. All references to that resource will be automatically modeled only at that pipeline stage. This is particularly useful for modeling shared encoding bits (typically for resource pools).  The syntax for that looks like:


```
    resource(E1) my_pool { res1, res2, res3 };
```


where E1 is the name of a pipeline phase.  The resource “my\_pool” (and each of its elements) is always modeled to be reserved in pipeline phase E1.


#### **Using Resources**

Resource references appear in several contexts.  They are used in all template instantiations to specialize architecture templates (functional units, subunit, or latency templates) and are ultimately used in latency rules to describe pipeline behaviors. These will be described later in the document.

When used to specialize template instances, resource references have the following grammar:


```
    resource_ref            : IDENT ('[' range ']')?
                            | IDENT '.' IDENT
                            | IDENT '[' number ']'
                            | IDENT ('|' IDENT)+
                            | IDENT ('&' IDENT)+ ;
```


Some examples of resource uses in functional unit instantiations, subunit instantiations, latency instantiations, and latency reference rules:


```
some_resource           // reference a single resource or an entire group/pool    
some_resource_pool[1]   // use a specific member from an unnamed pool.
register_ports[6..9]    // select a subset of unnamed pooled resources.
group.xyzzy             // select a single named item from a group.
res1 | res2 | res3      // select one of these resources
res6 & res7 & res8      // select all of these resources
```


References in latency reference rules have additional syntax to support the allocation of resources from groups and pools:


```
    latency_resource_ref    : resource_ref ':' number (':' IDENT)?
                            | resource_ref ':' IDENT (':' IDENT)?
                            | resource_ref ':' ':' IDENT
                            | resource_ref ':' '*'
                            | resource_ref ;
```



##### **Allocating Grouped and Pooled Resources**

Latency references allow you to optionally manage allocation of pooled resources, as well as specifying the significant bits of operands whose values can be shared with other instructions.

A reference of the form:


```
	some_resource_pool:1
```


indicates that a reference needs one element from a group/pooled resource associated with a latency reference. A reference of the form:


```
	some_resource_pool:2
```


indicates that the reference needs 2 (or more) _adjacent_ elements from a pooled resource associated with a latency reference.  A reference of the form:


```
	some_resource_pool:*
```


indicates that a reference needs _all _elements from a resource group or pool. Note that grouped resources can only use :1 and :\*.

A reference of the form:

	`some_resource_pool:size`

indicates an operand reference that requires some number of resources from the resource pool.   The number of resources needed is specified in the “size” attribute of the associated operand type. This enables us to decide at compile time how many resources to allocate for an instruction’s operand based on its actual value.  For example, large operand constant values may require more resources than small constants, while some operand values may not require any resources. There’s a specific syntax for describing these attributes in derived operand definitions (described earlier).

In the examples above, if the resource has shared bits associated with it (it’s shareable by more than one instruction), the entire contents of the operand are shared. In some cases, only part of the operand’s representation is shared, and we can can specify that with the following reference form:

	`some_resource_pool:size:mask`

This indicates that the associated operand’s “mask” attribute indicates which of the operand bits are sharable.  Finally, we can use a share-bits mask without allocation:

	`some_resource_pool::mask`

This reference utilizes the resource - or an entire pool - and uses the operand’s “mask” attribute to determine which bits are shared with other references.

We will describe how these references work when we describe latency rules.


### **Defining a Processor Family**

A TableGen description describes a family of processors, or subtargets, that share instruction and register definitions. Information about instruction behaviors are described with Schedules and Itineraries. The MDL also uses common instruction and register descriptions, scraped from TableGen, and adds first-class descriptions of CPUs, functional units, and pipeline modeling.

In an MDL CPU description, a CPU is described as an explicit set of functional units.  Each functional unit is tied to a set of subunits, and subunits are in turn explicitly tied to instruction definitions and pipeline behaviors.  There are two approaches for associating subunits with functional units, and the choice of which one to use is dependent on the attributes of the architecture you’re describing:



1. Subunit templates specify (either directly or through Latencies) which functional units they use, or
2. You define functional unit templates that specify exactly which subunits they use.

More detail on this below.


#### **Method 1: SuperScalar and Out-Of-Order CPUs**

Fully protected pipelines, forwarding, out-of-order issue and retirement, imprecise micro-operation modeling, and dynamic functional unit allocation make this class of 

CPUs difficult to model_ precisely._  However, because of their dynamic nature, precise modeling is both impossible and unnecessary.  But it is still important to provide descriptions that enable scheduling heuristics to understand the relative temporal behavior of instructions.

This method is similar to the way Tablegen “Schedules” associate instructions with a set of ReadWrite resources, which are in turn associated with sets of ProcResources (or functional units), latencies and micro-operations. This approach works well for superscalar and out-of-order CPUs, and can also be used to describe scalar processors.

The upside of this method is that you don’t need to explicitly declare functional unit templates.  You simply declare CPU instances of the functional units you want, and the MDL compiler creates implicit definitions for them.

The downside of this method is that you can’t specialize functional unit instances, which in turn means you can’t specialize subunit instances, or associated latency instances.  Fortunately, specialization generally isn’t necessary for this class of CPUs.  It would also be difficult to use this method to describe a typical VLIW processor (which is why we have method 2!).

We generally describe this as a “bottoms-up” approach (subunits explicitly tying to functional unit instances), and is the approach used by the Tablegen scraper (tdscan) for “Schedule-based” CPUs.


#### **Method 2: VLIWs, and everything else**

This method is appropriate for machines where we must provide more information about the detailed behavior of an instruction so that we can correctly model its issuing and pipeline behavior. It is particularly important for machines with deep, complex pipelines that _must_ be modeled by the compiler.  It has a powerful, flexible user-defined resource scheme which provides a lot more expressiveness than either “Schedules” or “Itineraries”. 

In this method, a functional unit instance is an instantiation of an _issuing_ functional unit, which is more typical of scalar and VLIW CPUs.  In the common case where different instances of a functional unit have different behaviors, we can easily model that using functional unit, subunit, and latency instance specialization, and more detailed latency rules. 

This approach allows a very high degree of precision and flexibility that's not available with method 1.  Its strictly more expressive than the first method, but much of that expressiveness isn’t required by superscalar CPUs.

We describe this as a “tops-down” approach (explicit functional unit template definitions

assert which subunits they support).  This is the method tdscan uses when scraping information about itineraries.


#### **Schema of a Full Processor Family Description**

 By convention, a description generally describes things in the following order (although the order of these definitions doesn’t matter):



*   Definition of the family name.
*   Describe the pipeline model(s).
*   Describe each CPU (subtarget) in terms of functional unit instances.
*   Describe each functional unit template in terms of subunit instances (tops-down approach)
*   Describe each subunit template type. A subunit represents a class of instruction definitions with similar execution behaviors, and ties those instructions to a latency description.
*   Describe each latency in terms of operand and resource references.

We will describe each of these items in more detail.  A machine description for a target has the following general schema: (a full syntax is provided in Appendix A)


```
    <family name definition>
    <pipeline phase descriptions>
    <global resource definitions>
    <derived operand definitions>

    // Define CPUs
    cpu gen_1 { 
       <cpu-specific resource definitions>
       <functional unit instance>
       <functional unit instance>
       …
    } 
    cpu gen_2 { … }
    …

	// Define Functional Unit Template Definitions (Tops-down approach)
    func_unit a_1(<functional unit parameters>) { 
       <functional-unit-specific resource and port definitions>
       <subunit instance>
	   <subunit instance>
	   …
}
    func_unit b_1(…) { … } 
    …

    // Define Subunit Template Definitions
    subunit add(<subunit parameters>) {
       <latency instance>
       <latency instance>
       …
    }
    subunit mul(…) { … }
    …

    // Latency Template Definitions
    latency add(<latency parameters>) {
       <latency reference>
       <latency reference>
       …
    }
    latency mul(…) { … }
	…

    // Instruction information scraped from Tablegen description
    <register descriptions>
    <register class descriptions>
    <operand descriptions>
    <instruction descriptions> 
```



##### **Bottoms-up vs Tops-down CPU Definition Schemas \
**

In the “tops-down” schema, we define CPUs, which instantiate functional units, which instantiate subunits, which instantiate latencies.  At each level of instantiation, the object (functional unit, subunit, latency) can be specialized for the context that it’s instantiated in.  We think of this as a “top-down” definition of a processor family. We provide detailed descriptions for each functional unit template, which we can specialize for each instance.

However, for many processors, this specialization is unnecessary, and the normal schema is overly verbose. For these kinds of processors, we can use the “bottoms-up” schema.

In this schema, the MDL compiler _implicitly_ creates functional unit and latency templates:



*   A CPU definition specifies which functional units are used in the normal syntax.
*   Subunits directly implement latency rules inline (rather than instantiate a latency template), including an explicit functional unit instance that they can execute on.

Here’s an example of this kind of bottom-up description:


```
    cpu dual_cpu {
    	func_unit ALU alu1();     // a "my_alu" functional unit, named "alu1"
    	func_unit ALU alu2();     // a "my_alu" functional unit, named "alu2"
    }
    subunit alu2() {{ def(E2, $dst); use(E1, $src); fus(ALU, 3); }}
    subunit alu4() {{ def(E4, $dst); use(E1, $src); fus(ALU, 7); }}
```


	`subunit alu7() {{ def(E7, $dst); use(E1, $src); fus(ALU, 42); }}`

Note that we don’t explicitly define the ALU functional unit template, but it is instantiated (twice) and used in three subunit/latency templates. Similarly, we don’t explicitly define the three latency templates.  Both the functional unit template and the latency templates are implicitly created in the MDL compiler. 

While this schema is much more compact, neither the functional units nor the subunits/latencies can be specialized. This is an appropriate approach for scalar and superscalar processors, and is used by tdscan for CPUs that use Tablegen Schedules.


#### **Specifying the Family Name**

A family name must be specified that ties the description to the LLVM name for the processor family.  It has the following grammar:


```
family_name        : 'family' IDENT ';' ;
```



#### **Pipeline Definitions**

We don’t explicitly define instruction “latencies” in the MDL. Instead, we specify when instructions’ reads and writes happen in terms of pipeline phases.  From this, we can calculate actual latencies. Rather than specify pipeline phases with numbers, we provide a way of naming pipeline stages, and refer to those stages strictly by name. A pipeline description has the following grammar:


```
    pipe_def           : protection? 'phases' IDENT '{' pipe_phases '}' ';'? ;
    protection         : 'protected' | 'unprotected' | 'hard' ;
    pipe_phases        : phase_id (',' phase_id)* ;
    phase_id           : '#'? IDENT ('[' range ']')? ('=' number)? ;
```


 For example:


```
phases my_pipeline { fetch, decode, read1, read2, ex1, ex2, write1, write2 };
```


We typically define these in a global phase namespace, and they are shared between CPU definitions. All globally defined phase names must be unique. However, each CPU definition can have private pipeline definitions, and names defined locally override globally defined names.

You can define more than one pipeline, and each pipeline can have the attribute “protected”, “unprotected”, or “hard”.  “Protected” is the default if none is specified.


```
protected phases alu { fetch, decode, ex1, ex2 };
unprotected phases vector { vfetch, vdecode, vex1, vex2 };
hard phases branch { bfetch, bdecode, branch };
```


A “protected” latency describes a machine where the hardware manages latencies between register writes and reads by injecting stalls into a pipeline when reads are issued earlier than their inputs are available, or resources are oversubscribed (pipeline hazards). Most modern general purpose CPUs have protected pipelines, and in the MDL language this is the default behavior.

An “unprotected” pipeline never inserts stalls for read-after-writes or pipeline hazards. In this type of pipeline, reads fetch whatever value is in the register (in the appropriate pipeline phase).  A resource conflict (hazard) results in undefined behavior (ie, the compiler must avoid hazards!). In this model, if an instruction stalls for some reason, the entire pipeline stalls. This kind of pipeline is used in several DSP architectures.

A “hard” latency typically describes the behavior of branch and call instructions, whose side effect occurs at a particular pipeline phase.  The occurrence of the branch or call always happens at that pipeline phase, and the compiler must accommodate that (by inserting code in the “delay slots” of the branch/call).

You can define multiple stages as a group - the following rule is equivalent to the first example above.

**<code>phases alu { fetch, decode, read[1..2], ex[1..2], write[1..2] };</code></strong> \


Like C enumerated values, each defined name is implicitly assigned an integer value, starting at zero and increasing sequentially, that represents its integer stage id.  You can explicitly assign values to pipeline phases, as in C, with the syntax  “`phase=value`”. You can also explicitly assign sequential values to a range, by using the syntax `"name[2..5]=value`”. 

Finally, there is specific syntax to annotate the first “execute” phase in a pipeline spec, using the ‘#’ syntax:


```
phases my_pipeline { fetch, decode, #read1, read2, ex1, ex2, write1, write2 };
```


This indicates that “read1” is the first execute stage in the pipeline, which serves as the “default” phase for any operand that isn’t explicitly described in a latency rule.


#### **CPU Definitions**

In the definition of a single CPU/subtarget, we specify a high-level description of the processor:



*   CPU-specific resource definitions
*   Specification of issue slots and issue-slot usage.
*   Specialized instances of available functional units, and/or clusters of functional units.
*   CPU-specific pipeline definitions.

Note that a CPU definition does not attempt to describe the pipeline behavior of functional units, but only specifies which functional units are implemented. The _behavior_ of functional units, and instructions that run on them, are explicitly described in the functional unit _templates._ 

Grammar:


```
    cpu_def                 : 'cpu' IDENT ('(' STRING (',' STRING)* ')')?
                                 '{' cpu_stmt* '}' ';'? ;
    cpu_stmt                : pipe_def
                            | resource_def
                            | reorder_buffer_def
                            | issue_statement
                            | cluster_instantiation
                            | func_unit_instantiation
                            | forward_stmt ;

    reorder_buffer_def      : 'reorder_buffer' '<' number '>' ';' ;

    cluster_instantiation   : 'cluster' IDENT '{' cluster_stmt+ '}' ';'? ;
    cluster_stmt            : resource_def
                            | issue_statement
                            | func_unit_instantiation
                            | forward_stmt ;

    issue_statement         : 'issue' '(' IDENT ')' name_list ';' ;

func_unit_instantiation : 'func_unit' func_unit_instance func_unit_bases*
                IDENT '(' resource_refs? ')'
                          		('->' (pin_one | pin_any | pin_all))?  ';' 

func_unit_instance      : IDENT ('<>' | ('<' number '>'))?  
    func_unit_bases         : ':' func_unit_instance

    pin_one                 : IDENT ;
    pin_any                 : IDENT ('|' IDENT)+ ;
    pin_all                 : IDENT ('&' IDENT)+ ;
```


The overall schema of a CPU definition looks like this:


```
    cpu gen_1 { 
       <cpu-specific resource definitions>
	   <cpu-specific issue definitions>
	   <cpu-specific pipeline definitions>
       <cluster definition or functional unit instance>
       <cluster definition or functional unit instance>
       <optional forwarding info>
       …
    }
```


and a cluster definition has the schema:

	`cluster xyz { `


```
       <cpu-specific resource definitions>
	   <cpu-specific issue definitions>
       <functional unit instance>
       <functional unit instance>
       <optional forwarding info>
       …
    } 
```


Below are some examples of increasingly complex CPU definitions.


##### **Simple Scalar CPU Definition**

In the simplest case, an empty CPU indicates a processor with no specific functional unit information. We assume a serial execution of instructions, with “default” latencies:


```
    cpu simplest_cpu { }
```


A single-alu CPU that has a scheduling model looks like this:


```
    cpu simple_cpu {
    	func_unit my_alu alu();      // a "my_alu" functional unit, named "alu"
    }
```


A slightly more complex example is a CPU that is single-issue, but has more than one execution pipeline:


```
    cpu dual_alu_cpu {
    	issue slot0;                 // a single issue pipeline
    	func_unit my_alu alu1();     // a "my_alu" functional unit, named "alu1"
    	func_unit my_alu alu2();     // a "my_alu" functional unit, named "alu2"
    }
```



##### **Multi-Issue CPUs**

Here’s an example of a 2-issue processor with two identical functional units:


```
    cpu dual_issue_cpu {
    	func_unit my_alu alu1();    // a "my_alu" functional unit, named "alu1"
    	func_unit my_alu alu2();    // a "my_alu" functional unit, named "alu2"
    }
```


Processors commonly have functional units with different capabilities - memory units, multipliers, floating point units, etc. The following is a four-issue CPU with 4 different types of functional units.


```
    cpu quad_cpu {
    	func_unit int_math imath();    // a "int_math" functional unit
    	func_unit float_math fmath();  // a "float_math" functional unit
    func_unit memory mem();        // a "memory" functional unit
    func_unit branch br();         // a "branch" functional unit
    }
```



##### **Defining Issue Slots**

Multi-issue CPUs always have a constrained set of instructions they can issue in parallel.  For superscalar, OOO processors this is generally tied to the number of issue pipelines that are available.  For VLIW, issue slots map directly to encoding bits in a parallel instruction.  In the MDL, you can explicitly define issue slots.  An example:


```
    cpu tri_cpu {
    	issue slot0, slot1;
    	func_unit my_alu alu1();     // a "my_alu" functional unit, named "alu1"
    	func_unit my_alu alu2();     // a "my_alu" functional unit, named "alu2"
    	func_unit my_alu alu3();     // a "my_alu" functional unit, named "alu3"
    }
```


In this example, we have 3 functional units, but only two issue slots.  So any of the three functional units can issue in either issue slot, but only two can be issued in parallel.

When issue slots are not specified, each functional unit runs in its own dedicated issue slot.


##### **Reservation of Issue Slots**

In VLIW architectures (in particular), some functional units may be “pinned” to a specific set of issue slots, or use multiple issue slots in some cases.  We provide syntax for specifying this:


```
    cpu three_issue_quad_cpu {
    	issue s0, s1, s2;
    	func_unit int_math alu1() -> s0;         // alu1 must issue in s0
    	func_unit float_math alu2() -> s1 | s2;  // alu2 must be in s1 or s2
    func_unit memory alu3() -> s0 & s1;      // alu3 uses both s0 and s1
    func_unit branch br();                   // branches can run in any slot
    }
```



##### **SuperScalar and Out-Of-Order CPUs**

In general, the overall approach for defining superscalar CPUs is quite different from other CPU types.   This class of architecture requires information about the size of the reorder buffer, and details about queues for each functional unit. Actual functional unit utilization is described in latency or subunit rules, which can specify exactly which functional units are used.

Functional units can be unreserved (like alu1, below), which means that an instruction or micro-operation that runs on that unit doesn’t actually use that specific resource.  A functional unit can have a single-entry queue - in which case it is unbuffered - or a specific size queue. 


```
    cpu three_issue_superscalar_cpu {
    	issue s0, s1, s2;
          reorder_buffer<20>;        		// the reorder buffer is size 20
    	func_unit int_math<> alu1();		// alu1 is unreserved
    	func_unit float_math<10> alu2();		// alu2 has 10 queue entries
    func_unit memory<20> alu3();		// alu3 has 20 queue entries
    func_unit branch br();			// branch has a single entry
    }
```



##### **Parameterized/Specialized Functional Unit Instances**

A functional unit template can be parameterized with register classes and resource references so that each instance of that functional unit template can be specialized for a specific context. The actual use of these parameters is specified in the functional unit template, explained in the following sections.  This section describes template specialization parameters.

A **register class parameter** asserts that the functional unit instance may impose a register constraint on instructions that execute on it. This constraint is an addition to the register class constraints specified by an instruction’s operand definitions. This enables us to model functional units that are connected to a subset - or a partition - of a register file. It can also be used to describe functional-unit-local register files. Finally, it can disqualify instructions from running on a functional unit if they have register operands or operand constraints that are incompatible with the functional unit constraints.


```
    register r[0..31];
    register_class ALL { r[0..31] };
    register_class LOW { r[0..15] };
    register_class HI  { r[16..31] };

    cpu my_cpu {
    	func_unit my_alu alu0(LOW);    // instructions use r0..r15
    	func_unit my_alu alu1(HI);     // instructions use r16..31
    }
    instruction add(ALL dst, ALL src1, ALL src2) { … }
```


A **resource parameter** indicates that instructions that execute on the functional unit may use that resource or a member of a resource pool. This is generally used to specify how shared resources are used across functional unit instances. 


```
    cpu my_cpu {
    	resource shared_thing;                   // a single shared resource
    	resource reg_ports { p1, p2, p3 };       // three associated resources
    	resource shared_stuff[20];               // 20 associated resources

    	func_unit math alu0(shared_thing);       // share a named resource
    	func_unit math alu1(reg_ports.p1);       // share one member of a group
    	func_unit math alu2(shared_stuff[12]);   // share one member of a pool

    	func_unit mem mem0(reg_ports);           // share an entire group
    	func_unit mem mem1(shared_stuff);        // share an entire pool
    	func_unit mem mem2(shared_stuff[3..14]); // share part of a pool
    }
```



##### **Functional Unit Clusters**

A processor definition can include named “clusters” of functional units. Each cluster can define local resources, and define its own issue rules.  The purpose of clusters is primarily as a syntactic convenience for describing processors with functional unit clusters.  An example:

	**<code>cpu my_cpu {</code></strong>


```
		cluster A {
			issue a, b;
			func_unit my_alu alu1();
    	func_unit my_alu alu2(); 
    		func_unit my_alu alu3();
    }
    cluster B {
			issue a, b;
			func_unit my_alu alu1();
    	func_unit my_alu alu2(); 
    		func_unit my_alu alu3();
    }
    }


```


This describes a 4-issue machine, where 2 instructions can  be issued on each cluster per cycle.


##### **Defining Compound Functional Unit Instances**

Its often convenient to define “compound” functional unit instances as collections that include 2 or more “component” units.  A compound unit includes all the capabilities of its component units.  Each component can specify its own reservation queue size.  


```
    cpu compound_units {
    	Issue s0, s1, s2;
    	func_unit int_math<5>:load<6> alu1();
    	func_unit int_math<5>:store<3> alu2();
    func_unit float_math<20>:branch<2> alu3();
    func_unit misc<30> alu4();	
    }
```


This construct is similar to the “super-unit” concept in tablegen.  Only one component of a compound functional unit can be used per cycle. In the above example, “alu3” is the only unit that supports floating point math or branches.  Consequently those operations can’t be issued in parallel.  Similarly, you can issue two integer math operations in parallel, but only if you’re not also issuing a load or store.

Currently, we don’t support specialization parameters on compound functional unit instances. However, you can define functional unit templates with base units, and this provides similar capability.


##### **Associating a CPU definition with an LLVM subtarget**

A cpu definition can be directly associated with one or more LLVM subtargets, for example:

	**<code>cpu SiFive7 ("sifive-7-series", "sifive-e76", "sifive-s76", "sifive-u74") { …</code></strong>

At compile time, we can select which CPU definition to use based on normal target-selection command-line options.


##### **Modeling Forwarding**

Forwarding is modeled by describing a forwarding network between functional units. The necessity of the concept of a “forwarding” network implies that such networks aren’t fully connected or uniform.

Grammar:


```
forward_stmt            : 'forward' IDENT '->'
            	                         forward_to_unit (',' forward_to_unit)* ';' ;
forward_to_unit         : IDENT ('(' snumber ')')?  ;
```


Example:


```
	forward my_alu -> my_adder(1), my_load(2);
```


In a forwarding specification, a unit name can be a functional unit instance name, a functional unit group name, or a functional unit template name.  When using template or group names, all members of the group, or all instances of the specified template type, are implicitly referenced.

For many processors, late functional unit assignment creates a phase-ordering problem in the compiler. Similarly, runtime functional unit assignment implies that we can’t necessarily know if a value will be forwarded or not.  Unless we know with certainty the functional unit assignments for two instructions, we can’t always tell if there is a forwarding path between the two instructions.

This isn’t necessarily a problem for downstream analysis tools, which work with fully scheduled code where all the functional units may have been determined by the compiler. There are several cases that we handle separately:

Case 1: two instructions are both tied to specific functional units: in this case, we can fully determine whether forwarding is supported between the two functional units.

Case 2: two instructions are tied to two sets of functional units (set A and set B) and all functional units in A are forwarded to all functional units in B.  In this case, we can also determine whether forwarding is supported between the two instructions. (We don’t attempt to manage this today.)

Case 3: Same as Case 2, but not all members of A are forwarded to B.  In this case, the compiler could use a probability of forwarding, perhaps.

Case 4: Same as Case 3, but there is no forwarding between A and B.

Note that case 3 is quite common, and can be mitigated if the compiler uses a pass to pre-constrain the sets of functional units each instruction uses. This is quite common in compilers for clustered architectures - a pre-scheduling pass chooses a cluster for each instruction, which effectively constrains the functional units each instruction can run on, and often improves the chances for forwarding between instructions.

Indeed the most common case currently modeled in tablegen files is a functional unit forwarding to itself or the superunit of itself, or a functional unit group forwarding to itself. 

In short, there are architectural cases that cannot be modeled precisely, and there are cases where we simply need a heuristic.  We provide the hooks necessary for a compiler to provide the heuristic based on the existing model.

Note: there is a philosophical question of whether we should provide best case or worst case latencies when the forwarding cannot be statically predicted.  Generally, we believe that worst case latencies are better than best case latencies, simply because too-short latencies can produce code which occasionally (or always) stalls. On the other hand, overestimating the latency produces schedules where a pair of dependent instructions _tend _to be scheduled far enough apart to avoid stalls. In effect, schedulers will separate instructions by the requested latency only when there’s other useful work to do.  Otherwise, there’s no reason to separate them - the stall is inevitable.


#### **Functional Unit Template Definitions**

A functional unit template describes, abstractly, what operations can be performed on any instance of the unit, and how those operations use the template parameters - register classes and resource references. An abstract set of operations is represented by a subunit instance, which represents a set of instructions with similar behavior in terms of functional unit usage, resource usage, and register classes. Functional unit templates are defined in their own private namespace.

Functional unit templates are similar to C++ templates in that each instantiation in CPU definitions creates a specialized instance of the functional unit based on the template parameters - register classes and resources.

For superscalar processors, it's not necessary to specify explicit templates for each functional unit used in a CPU description.  The MDL compiler instantiates these automatically depending on how the functional units are referenced in latency templates, tying functional units automatically to their associated subunits.   (The implication of this is that implicitly defined templates cannot be parameterized.)

A functional unit template has the following grammar:


```
    func_unit_template      : 'func_unit' IDENT base_list
                                    '(' func_unit_params? ')'
                                    '{' func_unit_template_stmt* '}' ';'? ;

    func_unit_params        : fu_decl_item (';' fu_decl_item)* ;
    fu_decl_item            : 'resource' name_list
                            | 'register_class' name_list ;

    func_unit_template_stmt : resource_def
                            | port_def
                            | connect_stmt
                            | subunit_instantiation ;

    port_def                : 'port' port_decl (',' port_decl)* ';' ;
    port_decl               : IDENT ('<' IDENT '>')? ('(' resource_refs ')')? ;
    connect_stmt            : 'connect' IDENT
                                 ('to' IDENT)? ('via' resource_refs)? ';' ;

    subunit_instantiation   : (name_list ':')? subunit_statement
                            | name_list ':' '{' subunit_statement* '}' ';'? ;

    subunit_statement       : 'subunit' subunit_instance (',' subunit_instance)* ';' ;
    subunit_instance        : IDENT '(' resource_refs? ')' ;
```


The general schema of a functional unit template looks like this: \



```
    func_unit a_1 [: <base_units>] (<functional unit parameters>) { 
       <functional-unit-specific resource and port definitions>
       <subunit instance>
	   <subunit instance>
	   …
}
```



##### **Simplest Functional Unit Template Definition**

The simplest example of a functional unit template would define a functional unit that has no parameters, and implements a single subunit:

	**<code>func_unit simple() {</code></strong>


```
		subunit xyzzy();
}
```


In this case, any instruction that is defined to use the subunit “xyzzy” can run on this functional unit. This template doesn’t impose any additional constraints on those instructions, and no shared resources are used.


##### **Defining Functional Unit Resources**

A functional unit template can locally define resources which represent hardware resources tied to _each instance_ of the functional unit.  These can be used to specialize subunit instances:

	**<code>func_unit unit_with_local_resources() {</code></strong>


```
		resource my_resource;
		resource my_pooled_resource[4];

		subunit add(my_resource, my_pooled_resource[0..1]);
	subunit subtract(my_resource, my_pooled_resource[2..3]);
	subunit multiply(my_pooled_resource);
}
```


In this example, the functional unit supports 3 classes of instructions (add, subtract, multiply), and passes slightly different local resources to each. Each instance of this functional unit has an independent set of resources (my\_resource, my\_pooled\_resource).

Importantly: functional-unit-local resources which are used for multiple cycles can be used to model non-pipelined functional units - i.e.units which are reserved for some number of cycles.


##### **Defining “Port” Resources**

A port is a resource type that explicitly binds a named register class with a resource reference. A port is used to specialize subunit instances, and adding functional-unit-specific register constraints on instructions associated with the subunit. 

A port definition has the general form:

	**<code>'port' <port_name> ('<' <register_class_name> '>')? ('(' resource_ref ')')? ;</code></strong>

When a port is tied to more than one resource, any references to that port refer to all of the associated resources.  Some examples:

	**<code>port port_a <GPR>;           // port_a tied to GPR regs</code></strong>


```
	port port_b <LOW> (res1);    // port_b tied to LOW regs and res1
	port port_c (pool[0..4]);    // port_c tied to pool[0..4]
```


You can also use a “connect” statement to tie a port to register classes and resources:

	**<code>'connect' <port_name> 'to' <register_class_name> 'via' resource_ref ;</code></strong>

The following is equivalent to the above definition of “port\_b”:

	**<code>port port_b;</code></strong>


```
connect port_b to LOW via res1;
```


This syntax could potentially be used to connect a port to more than one constraint/resource set, but this capability isn’t currently supported, and this syntax may be deprecated.

Ports can be used to specialize subunit and latency instances, described in subsequent sections.


##### **Using Template Parameters**

Resource parameters can be used exactly like locally defined resources to specialize subunit instances. Register class parameters are used to define ports. Resource parameters can refer to a single resource, a pool of resources, or a group of resources.

Here is an example subunit instance:


```
	subunit adder(res, porta, res2, portc);
```


The parameters refer to resources (or ports) defined in the functional unit, cluster, cpu, or globally. The resource parameters themselves can include constrained versions of the resources they refer to, in particular specifying a particular member or a subset of a pooled resource, for example:


```
	subunit load(pool1.member, pool2[5], pool3[2..4]);
```


A simple example of a full functional unit template definition:

	**<code>func_unit specialized(resource shared_pool; class regs) {</code></strong>


```
		resource my_resource;
		port my_port<regs> (shared_pool[3..5]);

		subunit load(my_resource, my_port);
	subunit store(my_port);
}
```



##### **Conditional Subunit Instances**

In a functional unit template, a subunit instance can be conditionally instantiated based on a predicate.  Predicates are simply names of the instantiating cpu definition and functional unit instance.  This allows us to specialize a functional unit instance based on how its instantiated, for example:

	**<code>cpu my_cpu {</code></strong>


```
		func_unit my_func xyzzy();
		func_unit my_func plugh();
	}
func_unit my_func() {
		resource pooled_resource[4];
		xyzzy: subunit add(pooled_resource[0..1]);
		plugh: subunit add(pooled_resource[2..3]);
}
```



##### **Using Base Functional Units**

Functional units tend to get more capable over generations of a processor, so we’d like a way to derive functional units from other functional units. A functional unit template can be defined to have a base functional unit, for example:


```
	func_unit base_func() { … }
	func_unit my_func : base_func() { … }
```


In this example, the template “my\_func” simply includes the definition of “base\_func” in its definition.  In effect, anything “base\_func” can do, “my\_func” can do.  The base functional unit definition must have the same leading parameters as the derived functional unit definition.

In effect, when you instantiate a based functional unit, you implicitly instantiate its bases and any subbases. This language feature allows us to easily extend functional unit definitions over processor generations.


##### **Defining functional unit groups**

When defining a superscalar CPU, its generally not necessary to provide a functional unit template definition for each functional unit, since latency rules specify which functional units are used by a subunit.  In this case, its helpful to be able to easily specify an arbitrary pool of functional units that can be used for an instruction.  So the MDL has a way to do that.


```
	func_unit_group          : 'func_group' IDENT ('<' number '>')? : name_list ;
```


For example:


```
    func_group MyGroup<42>  member1, member2, member3;
```


This defines a functional unit group with 3 members, and a single input queue of length 42.  These groups are used in latency rules to tie subunits to a pool of functional units.


#### **Subunit Template Definitions**

Subunits are used to link sets of instruction definitions to their pipeline behaviors and candidate functional units. Subunits appear in three contexts:



*   Each subunit template has a definition.
*   Functional unit templates instantiate subunits that they support.
*   Instructions can declare which subunits they are associated with. \


A subunit definition abstractly represents a set of instruction definitions that logically have the same behaviors:



*   When operands are read and written
*   What resources are used/held/reserved
*   What functional units they can issue on
*   What issue slots and/or encoding bits they use
*   What subtargets are supported

An instruction - or set of instructions - may behave differently between subtargets, and/or functional units, and/or issue slots. Subunit templates are therefore parameterized so that their instances can be specialized for the contexts in which they are instantiated, and they can in turn specialize their associated latency instantiations.

A subunit template definition has the following grammar:


```
    subunit_template        : 'subunit' IDENT su_base_list
                                 '(' su_decl_items? ')'
                                 (('{' subunit_body* '}' ';'?) |
                                  ('{{' latency_items* '}}' ';'? )) ;

    su_base_list            : (':' (IDENT | STRING_LITERAL))* ;
    su_decl_items           : su_decl_item (';' su_decl_item)* ;
    su_decl_item            : 'resource' name_list
                            | 'port'     name_list ;
    subunit_body            : latency_instance ;
    latency_instance        : (name_list ':')? latency_statement
                            | name_list ':' '{' latency_statement* '}' ';'? ;
    latency_statement       : 'latency' IDENT '(' resource_refs? ')' ';' ;
```


A subunit template has the following general schema:


```
    subunit add <base subunits> (<subunit parameters>) {
       <latency instance>
       <latency instance>
       …
    }
```


Latency instance instances (in subunit templates) have the following general forms:

	**<code>latency <latency_name> ( <subunit parameters> );</code></strong>


```
	<predicate> : latency <latency_name> ( <latency parameters> ); 
	<predicate> : { <latency statements> }
```


The optional predicate is a comma-separated list of names which refers to the CPU or functional unit the current subunit is instantiated in. This allows subunits to specify different latencies depending on the CPU or functional unit they are instantiated from. This is similar to the support in functional unit templates for conditional subunit instances. For example:


```
      cpu1, cpu3 : latency xyzzy(port1, port2, etc);
      alu7:        latency plugh(resource1, resource2, etc);
```


A subunit template can specify as many latency instances as needed - the resulting subunit is the union of all the valid latency templates.  This allows you to separate different classes of behaviors into different latency templates.  Since latency templates are also specialized, you can manage the separation in latencies. The typical practice is for a subunit to have a single latency instance.


##### Subunit Template Parameters

Subunit template parameters can be a mix of ports and resources, and are used to specialize a subunit for the context in which it is instantiated, for example:


```
    subunit add (resource A, B; port C) { … }
```


In general, these work exactly the same way functional unit templates are used.  They can be used as latency parameters to specialize latency instances.


##### Tying Instructions to Subunits

There are two ways to associate subunits to instructions:



*   Instructions can specify which subunits they can run on, or
*   Subunits can specify which instructions they support.

We discuss these two approaches below.


###### Subunits in Instructions

Subunits are associated with instruction definitions to...



*   Define each of their possible pipeline behaviors
*   Determine which functional units they can be issued on (if any!)
*   To provide functional-unit-specific register constraints to operand registers
*   To determine whether an instruction is valid for the selected architecture

Each defined instruction must specify at least one subunit that it is bound to. This is done in tablegen by introduction of a Subunit attribute on each instruction (or instruction class) definition.

We allow more than one subunit per instruction, which implies different instruction behaviors across CPUs or functional units.  In general, this isn’t necessary, since a subunit can specify different behaviors for different functional units and/or CPUs. So this is strictly a stylistic choice.


###### Subunit Bases

A subunit template definition can have one or more “bases”.  A base is either the name of another subunit, or a string representing a regular expression of instruction names. Bases tie a subunit to sets of instructions, either directly by instruction name, or transitively through their base subunits. A subunit does not need to have the same parameters as its bases, and does not inherit any latency information from its bases. 

This example ties the “add” subunit to any instruction with “ADD” as a name prefix, and also to any instructions tied to the “base\_add” subunit. 


```
    subunit add : "ADD*" : base_add() {...}
```


Subunit bases provide an alternate way of tying instructions to subunits without modifying the instruction definitions (where each instruction can tie itself to a set of subunits).  This effectively allows a single “base” subunit - and all of its associated instructions  - to have different latency behaviors for each target.


##### Shorthand Subunit Template Definitions

Often a subunit template simply specifies a single latency template instance, and the latency template may only be used in a single subunit template.  In that case, we have a shorthand that combines the latency template into the subunit template.  For example:


```
subunit load(resource a, b, c) {
	latency load(resource a, b, c);
}
latency load(resource a, b, c) { def(E1, $dst); use(E1, $src); … }
```


Can be alternatively expressed as:


```
subunit load(resource a, b, c) {{
	def(E1, $dst); use(E1, $src); … 
}}
```



#### **Latency Template Definitions**

A latency template specifies the detailed pipeline behavior for a class of instructions. The class of “client” instructions for a latency template is the set of instructions that use any subunit that instantiates the latency template.

Latency templates are specialized for the exact context they are instantiated in - so they are statically polymorphic: a single latency template instantiated in many contexts can describe many different behavior sets for a single instruction depending on the CPU, the functional unit instance, subunit instance, the latency instance, and the instruction itself.

Latency templates:



*   describe what happens at each stage of the execution pipeline in terms of register operands and resources used and reserved.
*   optionally imposes additional functional-unit-specific constraints on register operands.

A latency template definition has the following general schema:

	`latency <name> : base_latencies ( <parameters> ) {`


```
          <latency reference>
          <latency reference>
	   …
	}
```


Latency templates can be derived from other latencies, and take resources or ports as parameters. The body of the template is simply a set of latency references.

The full grammar:


```
    latency_template        : 'latency' IDENT base_list
                                 '(' su_decl_items? ')'
                                 '{' latency_items* '}' ';'? ;
    latency_items           : latency_refs
                            | micro_ops_statement ;
    latency_refs            : (name_list ':')?
                                   (latency_item | ('{' latency_item* '}' ';'?)) ;
    latency_item            : latency_ref
                            | conditional_ref
                            | fus_statement ;

    conditional_ref         : 'if' IDENT '{' latency_item* '}'
                                   (conditional_elseif | conditional_else)? ;
    conditional_elseif      : 'else' 'if' IDENT '{' latency_item* '}'
                                   (conditional_elseif | conditional_else)? ;
    conditional_else        : 'else' '{' latency_item* '}' ;

    latency_ref             : ref_type '(' latency_spec ')' ';' ;
    ref_type                : ('use' | 'def' | 'usedef' | 'kill' |
                               'hold' | 'res' | 'predicate' | 'fus') ;
    latency_spec            : expr (':' number)? ',' latency_resource_refs
                            | expr ('[' number (',' number)? ']')? ',' operand
                            | expr ',' operand ',' latency_resource_refs ;
    expr                    : '-' negate=expr
                            | expr ('*' | '/') expr
                            | expr ('+' | '-') expr
                            | '{' expr '}'
                            | '(' expr ')'
                            | IDENT
                            | number
                            | operand ;

    fus_statement           : 'fus' '(' (fus_item ('&' fus_item)*  ',')?
                                      snumber (',' fus_attribute)* ')' ';'

    fus_item                : IDENT ('<' (expr ':')? number '>')? ;
    fus_attribute           : 'BeginGroup' | 'EndGroup' | 'SingleIssue'
  	                        | 'RetireOOO' ;

    latency_resource_refs   : latency_resource_ref (',' latency_resource_ref)* ;
    latency_resource_ref    : resource_ref ':' number (':' IDENT)?
                            | resource_ref ':' IDENT (':' IDENT)?
                            | resource_ref ':' ':' IDENT      // no allocation
                            | resource_ref ':' '*'            // allocate all
                            | resource_ref ;
    operand                 : (IDENT ':')? '$' IDENT ('.' operand_ref)*
                            | (IDENT':')? '$' number
                            | (IDENT':')? '$$' number

    operand_ref             : (IDENT | number) ;
```



##### **Derived Latency Templates**

A latency template can be derived from one or more base latency templates.  Any hierarchy is allowed (except recursive), as long as the base template has the exact same leading parameters as the derived latency.  A base latency can be included more than once in the hierarchy - this doesn’t matter, since all occurrences of that base are identical (so duplicates are ignored):


```
	latency base1 (resource a) { … }
	latency base2 (resource a, b) { … }
	latency base3 : base1 (resource a) { … }
	latency my_latency : base2 : base3(resource a, b, c) { … }
```


In this example, my\_latency includes base1, base2, and base3. Deriving latency templates is a fairly common pattern: instruction classes often share _some_ behaviors, but not all.  So those shared behaviors can be put in a base latency template.  A common example is an instruction predicate, perhaps shared by all instructions.


##### **Latency References**

A latency reference statement describes a single operand reference and/or resource references in a specified pipeline stage. It references instruction operands _by name, _as well as resource and port parameters, and ties the operations to named pipeline phases.

Latency references have the following general form:


```
<operator> (<phase expression>, <operand specifier>, <ports/resources>);
```


where either the operand specifier or ports/resources may be omitted. A single reference statement asserts that an operand and resources are referenced in a specific pipeline phase for any instruction that this rule could apply to, ie: _any instruction that uses a subunit that instantiates this latency template._  Each aspect of a latency reference are described below.


###### **Operand and resource latency operators:**

There are 6 basic operator types in a latency reference:



*   use - read a register, and/or use a resource
*   def - write a register, and optional use of a resource
*   predicate - specifies which register operand is an instruction predicate
*   reserve - reserve a resource until a specific pipeline stage.
*   hold - hold issue until a resource is available for reservation.
*   fus - reserve a functional unit for a specified number of cycles, and/or a specified number of micro-ops needed for the instruction.

There are 3 additional operator types which are primarily used as shortcuts (these are currently parsed, but unimplemented in the llvm integration):



*   usedef - a use and a def of an operand (a shorthand syntax) 
*   kill - the register value is wiped and no value is defined (typically used in call instructions)
*   or - this is essentially a conditional def, but the instruction has no explicit predicate (useful for status-setting instructions).


###### **Phase Expressions**

The phase expression specifies the pipeline phase that the operation occurs in. The expression can refer directly to a defined phase name, or an expression based on a phase name:


```
	use(E7, $operand, res);    // use operand and res in cycle E7
	use(E7+5, $operand, res);  // use operand and res in cycle E7+5
```


An instruction may perform a reference at a cycle which is a function of immediate operands of the instruction instance.  For example: 


```
	use(E1 + $width - 12, $operand, res);
```


where “$width” is an immediate instruction operand. Its value is fetched from the instruction instance and used in the expression.  As with any operand specifier, if the client instruction doesn’t have an immediate operand named “width”, the rule is ignored for that instruction.

Phase expressions have a limited set of operators: +, -, \*, /, ().  Since latencies must be positive integers, we also provide a “floor” operator which converts negative expressions to 0. Simply enclose the expression in curly braces ({...}).


###### **Operand Specifiers**

The operand specifier has the same grammar as in tablegen, which allows you to specify an optional operand type, the operand name, and optional sub-operand names: 


```
    operand                 : (IDENT ':')? '$' IDENT ('.' operand_ref)*
                            | (IDENT':')? '$' number
                            | (IDENT':')? '$$' number

    operand_ref             : (IDENT | number) ;
```


Operand specifiers act as predicates for the validity of a reference for a particular instruction. Some examples:


```
	GPR:$dst         // an operand named "dst" with operand type GPR
	ADR:$dst         // an operand named "dst" with operand type ADR
	$dst             // an operand named "dst", with any operand type
```


`	opnd:$src.reg    // an operand named "src", type "opnd", suboperand "reg"` 

Because a latency could be specialized for many instructions which have different sets of operands, the operand specifier acts as a predicate for the application of a reference to a particular instruction. When the operand isn’t present in a client instruction, the latency reference is ignored for that instruction.  For example, you can differentiate on operand type:


```
	def(E5, GPR:$dst);
	def(E7, FPR:$dst);
```


In this example, instructions with a GPR dst operand write their results in cycle E5, while instructions with an FPR dst operand write their results in cycle E7.

Or you can differentiate based on the operand name:


```
	use(E2, $src1);              // most instructions have at least one src opnd
	use(E3, $src2);              // some instructions have 2 source operands
```


`	use(E4, $src3);              // and some instructions have 3!` \


Note that operands _can _be referenced by their index in an instruction’s operand list, but this is error-prone and this isn’t considered best practice because we can’t thoroughly check the validity of the index.  The syntax is simply “$<index>”.  Note that sub-operands often aren’t given names in tablegen, and must be referenced by index, for example: $src.1.  Unnamed variant operands (obviously) don’t have names, and are referenced by their position past the end of the operands defined for an instruction, ie “$$1”, “$$2”, etc.


###### **Resource References**

Any latency reference can include an optional set of resource references. These have slightly different semantics depending on the operator type (def/use/predicate/hold/reserve).

For “use”, “def”, and “predicate” statements, a set of resource references can be specified that are associated with the operand reference. As with all latency references, the operand must match an operand of the client instruction. If the reference is valid, the resource is “used” - for any of these operators - at the pipeline phase specified, unless the resource was defined with a specific phase. The “use” of the resource is equivalent to a single-cycle hold/reserve of that resource.  Some examples:


```
	use(E1, $src, my_res);       // use "my_res" at cycle E1
	def(E32, $dst, my_res);      // use "my_res" at cycle E32
```


For “hold” and “reserve” operations, the operand specifier is optional, and if present serves _only _as a predicate that indicates whether the reference is valid or not. However, at least one resource reference is required for these statements. A few examples:


```
	hold(E1, my_res);           // hold issue at E1 until resources are available
	res(E32, $dst, my_res);     // reserve resources up to cycle E32
```



##### **Conditional References**

Any reference in a latency rule can be conditional, using a predicate identifier.  The predicates are generally identical to LLVM predicates, and check an attribute of a client instruction. 

Conditional references can be nested, for arbitrarily complex references. These have the following general form:


```
if <predicate_name> { <set of refs> } 
else if <predicate_name> { <set of refs> }
else { <set of refs> } 
```



##### **Functional Unit and Micro-op References**

A latency rule can directly specify a set of functional units and how long they are used, as well as specifying the number of micro-ops required for the operation.  Each functional unit can optionally specify a pipeline “StartAt” cycle, which by default is the first execution phase.


```
	fus(13);                    // Instruction has 13 micro-operations.
fus(ALU, 2);                // use ALU for 1 cycle, 2 micro-operations.
	fus(ALU<3>, 1);             // use ALU for 3 cycles, 1 micro-operation.
	fus(ALU<E5:4>, 1);          // use ALU starting at E5 for 4 cycles.
	fus(ALU1<12>&ALU2<E12:30>&LOAD<E42:2>);    // use ALU1, ALU2, and LOAD
```


These statements allow a latency rule (or subunit) to tie a set of instructions to functional unit instances.  When there is more than one instance of the specified unit, or if the unit is declared to be a functional unit group, at compile time _one_ of those units is selected.  Likewise, if the unit is a subunit of one or more functional units, one of the “parent” functional units is selected.


### **Machine Description Compiler Artifacts**

What does all of this produce?

The primary artifact of the MDL compiler is a set of data that we associate with each instruction description in a targeted compiler.  For each instruction, at compiler-build time we produce a list of objects, each of which describe that instruction’s behavior on a single functional unit instance.  The instruction will have one of these objects for each functional unit instance that it can be scheduled on across all CPUs. These are written out as a set of auto-initialized collections of objects that are attached to instruction templates in the target compiler.

Each of these objects describe the behavior of each instruction cycle by cycle:



*   What operand’s registers it reads and write,
*   What register constraints are applied to operands,
*   What resources it uses, holds on, or reserves.
*   What explicit functional unit and issue slots it uses.
*   What pooled resources need to be allocated.

The other primary artifact is a set of objects and methods for managing the low-level details of instruction scheduling and register allocation.  This includes methods to build and manage resource pools, pipeline models, resource reservation infrastructure, and instruction bundling, all specialized for the input machine description.

As part of this effort, we will incrementally modify the LLVM compiler to alternatively use this information alongside of SchedMachineModel and Itinerary methodologies.



<p id="gdcalert1" ><span style="color: red; font-weight: bold">>>>>  GDC alert: inline image link here (to images/image1.png). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert2">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>> </span></p>


![alt_text](images/image1.png "image_tooltip")



### **Using the MDL Language in LLVM**

The proposed use case for the MDL language is as an alternative specification for the architecture description currently embodied in TableGen Schedules and Itineraries, particularly for architectures for which Schedules and Itineraries are not expressive enough.  It is explicitly _not_ the intent that it “replace TableGen”.  But we believe that the MDL language is a better language (vs Schedules and Itineraries) for a large class of accelerators, and can be used effectively alongside TableGen.

We’ve written a tool (TdScan) which extracts enough information from TableGen descriptions so that we can sync instruction definitions with architecture definitions. TdScan can also optionally scrape all of the Schedule and Itinerary information from a tablegen description and produce an equivalent\*\* MDL description.

So there are several possible MDL usage scenarios:



*   _Current: _Given a complete tablegen description with schedules or itineraries, scrape the architecture information and create an MDL description of the architecture every time you build the compiler.
*   _Transitional: _Scrape an existing tablegen description and keep the generated MDL file, using it as the architecture description going forward.
*   _Future (potentially): _when writing a compiler for a new architecture, write an MDL description rather than schedules and/or itineraries.

The general development flow of using an MDL description in LLVM looks like this: 



1. Write an architecture description (or scrape one from an existing tablegen description).
    1. Instructions, operands, register descriptions in .td files
    2. Microarchitecture description in .mdl files
2. Compile TD files with TableGen 
3. Use TdScan to scrape instruction, operand, and register information from tablegen, producing a .mdl file
4. Compile the top-level MDL file (which includes the scraped Tablegen information). This produces C++ code for inclusion in llvm.
5. Build LLVM.



<p id="gdcalert2" ><span style="color: red; font-weight: bold">>>>>  GDC alert: inline image link here (to images/image2.png). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert3">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>> </span></p>


![alt_text](images/image2.png "image_tooltip")



#### **TdScan**

To synchronize an MDL architecture description with llvm TableGen descriptions, we’ve written a tool which scrapes information that the MDL compiler needs from Tablegen files. In the general case, it collects basic information about registers, register classes, operands, and instruction definitions, and it produces an “mdl” file which can be processed by the MDL compiler to sync an architecture description to the tablegen descriptions of instructions.

For currently upstreamed targets that use Schedules or Itineraries, TdScan can also extract the whole architecture specification from the tablegen files, and produce an MDL description of the architecture. We’ve used this approach to prove out our llvm integration with upstreamed targets. The integration and testing of this is ongoing.


#### **Upstream Targets**

In general, upstream targets have no compelling need for MDL descriptions - the existing Schedules and/or Itinerary descriptions are field tested. However, there are a few benefits to using an MDL description for existing targets.  The primary benefit is that the MDL descriptions are typically quite a bit smaller, succinct, and (we believe) intuitive than the equivalent TableGen descriptions. 


<table>
  <tr>
   <td><strong>CPU</strong>
   </td>
   <td><strong>MDL Lines of Code</strong>
   </td>
   <td><strong>Tablegen Lines of Code</strong>
   </td>
  </tr>
  <tr>
   <td><strong>AArch64</strong>
   </td>
   <td>3927/4877
   </td>
   <td>28612
   </td>
  </tr>
  <tr>
   <td><strong>AMDGPU</strong>
   </td>
   <td>530/512
   </td>
   <td>440
   </td>
  </tr>
  <tr>
   <td><strong>ARM</strong>
   </td>
   <td>3486/3853
   </td>
   <td>10352
   </td>
  </tr>
  <tr>
   <td><strong>Hexagon</strong>
   </td>
   <td>3061/3061
   </td>
   <td>18743
   </td>
  </tr>
  <tr>
   <td><strong>Lanai</strong>
   </td>
   <td>54/54
   </td>
   <td>69
   </td>
  </tr>
  <tr>
   <td><strong>Mips</strong>
   </td>
   <td>874/967
   </td>
   <td>3003
   </td>
  </tr>
  <tr>
   <td><strong>PowerPC</strong>
   </td>
   <td>5103/5442
   </td>
   <td>4105
   </td>
  </tr>
  <tr>
   <td><strong>RISCV</strong>
   </td>
   <td>305/374
   </td>
   <td>3231
   </td>
  </tr>
  <tr>
   <td><strong>Sparc</strong>
   </td>
   <td>273/273
   </td>
   <td>123
   </td>
  </tr>
  <tr>
   <td><strong>SystemZ</strong>
   </td>
   <td>1643/1524
   </td>
   <td>9224
   </td>
  </tr>
  <tr>
   <td><strong>X86</strong>
   </td>
   <td>7263/9237
   </td>
   <td>30815
   </td>
  </tr>
</table>


\*\* Note: the MDL numbers are generated both with and without “index-based” references in subunit/latency rules, vs symbolic references.  These are typically 10-20% less lines of MDL description than when operand names are used, almost entirely due to operand name differences between instruction definitions (like “dest” vs “dst”, or “src1” vs “s1”).  However, the databases produced by the two approaches are virtually identical - albeit ordered differently.


#### **Syncing Instruction Information**

The MDL compiler needs 3 pieces of information from tablegen for each machine instruction:



1. The instruction opcode name
2. Each operand’s name, type, and order of appearance in an instruction instance
3. The name(s) of the subunit(s) it can run on.

Subunits are a new concept introduced with the MDL.  The normal approach is to modify each tablegen instruction description to explicitly specify subunit assignments, which become an additional instruction attribute.  The other approach is to use subunit template bases to use regular expressions to tie instructions to subunits (just like InstRW records).  

As part of the build process, we use a program (“tdscan”) which scrapes the instruction information - including the subunit information from a target’s tablegen files and generates information about the target’s instructions.  Tdscan allows us to stay in sync with changes to instruction definitions.


#### **Using the generated microarchitecture information in LLVM**

There are two classes of services that the MDL database and associated APIs provide:



*   Detailed pipeline modeling for instructions (for all processors, for all functional units) including instruction latencies calculations and resource usage (hazard management)
*   Parallel instruction bundling and instruction scheduling.

The tablegen scraper (tdscan) can correctly scan all upstreamed targets and generate correct instruction, operand, and register class information for all of them.

We can also extract high-level architecture information and generate correct MDL descriptions for all the upstreamed targets that have Schedules or Itineraries (AArch64, AMDGPU, AMD/R600, ARM, Hexagon, Lanai, Mips, PPC, RISCV, Sparc, SystemZ, X86).  Usually, the new architecture spec is dramatically simpler than the tablegen descriptions.

We provide code and libraries to do the following things - in machine-independent ways:



*   Calculate accurate instruction latencies.
*   A set of APIs to build and manage instruction bundles (parallel instructions), performing all the required legality checks and resource allocation based on information in the generated database.
*   Manage resource reservations and hazard management for an instruction scheduler.
*   Determine latencies between instructions based on resource holds and reservations.
*   Methods to query functional unit, issue slot, and resource assignments for a bundled/scheduled instruction.
*   Methods to query the set of all register uses and defs for an instruction instance, with accurate timing information.
*   Manage functional unit register forwarding.

There’s more we can do here, and a deeper integration with upstreamed LLVM is a long-term goal.


#### **Current Status of the LLVM Integration (briefly)**



*   We can generate MDL full architecture specs for all upstreamed targets, and properly represent and use all metadata associated with Schedules and Itineraries.
*   We’ve integrated the MDL methodology into LLVM’s build flow, so that you can select whether or not to include it at build time.  
*   The MDL database is (optionally, under command line control) used to properly calculate instruction latencies for all architectures.  Caveat: we don’t yet fully convert Itinerary and Schedule forwarding information, since the LLVM model for forwarding is fundamentally different from the MDL model, and the provided information is typically incomplete.
*   We’ve integrated the MDL-based bundle-packing and hazard management into all the LLVM schedulers, with the exception of the Swing scheduler, which is still in progress.
*   We’ve run all the standard tests, passing all but 190 (out of 93007 tests), with any performance deltas in the noise.

 


### **Appendix A: Full Language Grammar**

This may be slightly out of date.  The definitive Antlr4-based grammar is in llvm/utils/MdlCompiler/mdl.g4. 


```
architecture_spec       : architecture_item+ EOF ;
architecture_item       : family_name
                        | cpu_def
                        | register_def
                        | register_class
                        | resource_def
                        | pipe_def
                        | func_unit_template
                        | func_unit_group
                        | subunit_template
                        | latency_template
                        | instruction_def
                        | operand_def
                        | derived_operand_def
                        | import_file
                        | predicate_def ;

import_file             : 'import' STRING ;

family_name             : 'family' IDENT ';' ;

//---------------------------------------------------------------------------
// Top-level CPU instantiation.
//---------------------------------------------------------------------------
cpu_def                 : 'cpu' IDENT ('(' STRING (',' STRING)* ')')?
                             '{' cpu_stmt* '}' ';'? ;

cpu_stmt                : pipe_def
                        | resource_def
                        | reorder_buffer_def
                        | issue_statement
                        | cluster_instantiation
                        | func_unit_instantiation
                        | forward_stmt ;

cluster_instantiation   : 'cluster' IDENT '{' cluster_stmt+ '}' ';'? ;

cluster_stmt            : resource_def
                        | issue_statement
                        | func_unit_instantiation 
                        | forward_stmt ;

issue_statement         : 'issue' '(' IDENT ')' name_list ';' ;

func_unit_instantiation : 'func_unit' func_unit_instance func_unit_bases*
                IDENT '(' resource_refs? ')'
                          		('->' (pin_one | pin_any | pin_all))?  ';'

func_unit_instance      : IDENT ('<>' | ('<' number '>'))?  
func_unit_bases         : ':' func_unit_instance

pin_one                 : IDENT ;
pin_any                 : IDENT ('|' IDENT)+ ;
pin_all                 : IDENT ('&' IDENT)+ ;

//---------------------------------------------------------------------------
// A single forwarding specification (in CPUs and Clusters).
//---------------------------------------------------------------------------
forward_stmt            : 'forward' IDENT '->'
                                      forward_to_unit (',' forward_to_unit)* ';' ;
forward_to_unit         : IDENT ('(' snumber ')')? ;

//---------------------------------------------------------------------------
// Functional unit template definition.
//---------------------------------------------------------------------------
func_unit_template      : 'func_unit' IDENT base_list
                                '(' func_unit_params? ')'
                                '{' func_unit_template_stmt* '}' ';'? ;

func_unit_params        : fu_decl_item (';' fu_decl_item)* ;
fu_decl_item            : 'resource'  name_list
                        | 'register_class' name_list ;

func_unit_template_stmt : resource_def
                        | port_def
                        | connect_stmt
                        | subunit_instantiation ;

port_def                : 'port' port_decl (',' port_decl)* ';' ;
port_decl               : IDENT ('<' IDENT '>')? ('(' resource_ref ')')? ;
connect_stmt            : 'connect' IDENT
                             ('to' IDENT)? ('via' resource_ref)? ';' ;

//---------------------------------------------------------------------------
// Functional unit group definition.
//---------------------------------------------------------------------------
func_unit_group         : FUNCGROUP IDENT ':' name_list ';' ;

//---------------------------------------------------------------------------
// Definition of subunit template instantiation.
//---------------------------------------------------------------------------
subunit_instantiation   : (name_list ':')? subunit_statement
                        | name_list ':' '{' subunit_statement* '}' ';'? ;

subunit_statement       : 'subunit' subunit_instance (',' subunit_instance)* ';' ;
subunit_instance        : IDENT '(' resource_refs? ')' ;

//---------------------------------------------------------------------------
// Definition of subunit template definition.
//---------------------------------------------------------------------------
subunit_template        : 'subunit' IDENT su_base_list '(' su_decl_items? ')'
                             (('{' subunit_body* '}' ';'?) |
                              ('{{' latency_items* '}}' ';'? )) ;

su_decl_items           : su_decl_item (';' su_decl_item)* ;
su_decl_item            : 'resource' name_list
                        | 'port'     name_list ;

su_base_list            : (':' (IDENT | STRING_LITERAL))* ;

subunit_body            : latency_instance ;
latency_instance        : (name_list ':')? latency_statement
                        | name_list ':' '{' latency_statement* '}' ';'? ;
latency_statement       : 'latency' IDENT '(' resource_refs? ')' ';' ;

//---------------------------------------------------------------------------
// Latency template definition.
//---------------------------------------------------------------------------
latency_template        : 'latency' IDENT base_list
                             '(' su_decl_items? ')'
                             '{' latency_items* '}' ';'? ;

latency_items           : (name_list ':')?
                               (latency_item | ('{' latency_item* '}' ';'?)) ;

latency_item            : latency_ref
                        | conditional_ref
                        | fus_statement ;

//---------------------------------------------------------------------------
// Conditional references
//---------------------------------------------------------------------------
conditional_ref         : 'if' IDENT '{' latency_item* '}'
                               (conditional_elseif | conditional_else)? ;
conditional_elseif      : 'else' 'if' IDENT '{' latency_item* '}'
                               (conditional_elseif | conditional_else)? ;
conditional_else        : 'else' '{' latency_item* '}' ;

//---------------------------------------------------------------------------
// Basic references
//---------------------------------------------------------------------------
latency_ref             : ref_type '(' latency_spec ')' ';' ;

ref_type                : ('use' | 'def' | 'usedef' | 'kill' |
                           'hold' | 'res' | 'predicate') ;

latency_spec            : expr (':' number)? ',' latency_resource_refs
                        | expr ('[' number (',' number)? ']')? ',' operand
                        | expr ',' operand ',' latency_resource_refs ;

expr                    : '-' expr
                        | expr ('*' | '/') expr
                        | expr ('+' | '-') expr
                        | '{' expr '}'
                        | '(' expr ')'
                        | IDENT
                        | number
                        | operand ;

//---------------------------------------------------------------------------
// Shorthand for a reference that uses functional units.
//---------------------------------------------------------------------------
fus_statement           : 'fus' '(' (fus_item ('&' fus_item)* ',')?
                                  snumber (',' fus_attribute)* ')' ';'
                        ;
fus_item                : IDENT ('<' (expr ':')? number '>')? ;

fus_attribute           : 'BeginGroup' | 'EndGroup' | 'SingleIssue' | 'RetireOOO' ;

//---------------------------------------------------------------------------
// Latency resource references
//---------------------------------------------------------------------------
latency_resource_refs   : latency_resource_ref (',' latency_resource_ref)* ;

latency_resource_ref    : resource_ref ':' number (':' IDENT)?
                        | resource_ref ':' IDENT (':' IDENT)?
                        | resource_ref ':' ':' IDENT       // no allocation
                        | resource_ref ':' '*'             // allocate all members
                        | resource_ref ;

operand                 : (IDENT ':')? '$' IDENT ('.' operand_ref)*
                        | (IDENT ':')? '$' number
                        | (IDENT ':')? '$$' number

operand_ref             : (IDENT | number) ;

//---------------------------------------------------------------------------
// Pipeline phase names definitions.
//---------------------------------------------------------------------------
pipe_def                : protection? 'phases' IDENT '{' pipe_phases '}' ';'? ;
protection              : 'protected' | 'unprotected' | 'hard' ;
pipe_phases             : phase_id (',' phase_id)* ;
phase_id                : '#'? IDENT ('[' range ']')? ('=' number)? ;

//---------------------------------------------------------------------------
// Resource definitions: global in scope, CPU- or Datapath- or FU-level.
//---------------------------------------------------------------------------
resource_def            : 'resource' ( '(' IDENT ('..' IDENT)? ')' )?
                              resource_decl (',' resource_decl)*  ';' ;

resource_decl           : IDENT (':' number)? ('[' number ']')?
                        | IDENT (':' number)? '{' name_list '}' 
                        | IDENT (':' number)? '{' group_list '}' ;

resource_refs           : resource_ref (',' resource_ref)* ;

resource_ref            : IDENT ('[' range ']')?
                        | IDENT '.' IDENT
                        | IDENT '[' number ']'
                        | IDENT ('|' IDENT)+
                        | IDENT ('&' IDENT)+ ;

//---------------------------------------------------------------------------
// List of identifiers.
//---------------------------------------------------------------------------
name_list               : IDENT (',' IDENT)* ;
group_list              : IDENT ('|' IDENT)+
                        | IDENT ('&' IDENT)+ ;

//---------------------------------------------------------------------------
// List of template bases
//---------------------------------------------------------------------------
base_list               : (':' IDENT)* ;

//---------------------------------------------------------------------------
// Register definitions.
//---------------------------------------------------------------------------
register_def            : 'register' register_decl (',' register_decl)* ';' ;
register_decl           : IDENT ('[' range ']')? ;

register_class          : 'register_class' IDENT
                            '{' register_decl (',' register_decl)* '}' ';'?
                        | 'register_class' IDENT '{' '}' ';'? ;

//---------------------------------------------------------------------------
// Instruction definition.
//---------------------------------------------------------------------------
instruction_def         : 'instruction' IDENT
                            '(' (operand_decl (',' operand_decl)*)? ')'
                            '{'
                                ('subunit' '(' name_list ')' ';' )?
                                ('derived' '(' name_list ')' ';' )?
                            '}' ';'? ;

//---------------------------------------------------------------------------
// Operand definition.
//---------------------------------------------------------------------------
operand_def             : 'operand' IDENT
                             '(' (operand_decl (',' operand_decl)*)? ')'
                             '{' (operand_type | operand_attribute)* '}' ';'?
                        ;
operand_decl            : ((IDENT (IDENT)?) | '...') ('(I)' | '(O)')? ;

operand_type            : 'type' '(' IDENT ')' ';' ;

operand_attribute       : (name_list ':')? operand_attribute_stmt
                        | name_list ':' '{' operand_attribute_stmt* '}' ';'? ;
operand_attribute_stmt  : 'attribute' IDENT '=' (snumber | tuple)
                           ('if' ('lit' | 'address' | 'label')
                 ('[' pred_value (',' pred_value)* ']' )? )? ';' ;
pred_value              : snumber
                        | snumber '..' snumber
                        | '{' number '}' ;

//---------------------------------------------------------------------------
// Derived Operand definition.
//---------------------------------------------------------------------------
derived_operand_def     : 'operand' IDENT base_list  ('(' ')')?
                              '{' (operand_type | operand_attribute)* '}' ';'? ;

//---------------------------------------------------------------------------
// Predicate definition.
//---------------------------------------------------------------------------
predicate_def           : 'predicate' IDENT ':' predicate_op? ';' ;

predicate_op            : pred_opcode '<' pred_opnd (',' pred_opnd)* ','? '>'
                        | code_escape
                        | IDENT ;
code_escape             : '[' '{' .*? '}' ']' ;

pred_opnd               : IDENT
                        | snumber
                        | STRING_LITERAL
                        | '[' IDENT (',' IDENT)* ']'
                        | predicate_op
                        | operand ;

pred_opcode             : 'CheckAny' | 'CheckAll' | 'CheckNot' | 'CheckOpcode'
                        | 'CheckIsRegOperand' | 'CheckRegOperand'
                        | 'CheckSameRegOperand' | 'CheckNumOperands'
                        | 'CheckIsImmOperand' | 'CheckImmOperand'
                        | 'CheckZeroOperand' | 'CheckInvalidRegOperand'
                        | 'CheckFunctionPredicate' | 'CheckFunctionPredicateWithTII'
                        | 'TIIPredicate'
                        | 'OpcodeSwitchStatement' | 'OpcodeSwitchCase'
                        | 'ReturnStatement' | 'MCSchedPredicate' ;

//---------------------------------------------------------------------------
// Match and convert a number, a set of numbers, and a range of numbers.
//---------------------------------------------------------------------------
number                   : NUMBER ;
snumber                  : NUMBER | '-' NUMBER ;
tuple                    : '[' snumber (',' snumber)* ']' ;
range                    : number '..' number ;
```



### **Appendix B: Future Directions**


#### **Memory Hierarchy**

We need a first class representation of any compiler-managed memory hierarchy. 

Compiler-managed memory



*   Per level
    *   Size
    *   Addressable units
    *   Speed
    *   Latency
    *   Access method(s)
    *   Banking
    *   Sharing
*   Separate address spaces
    *   Code, Data, I/O, etc

Caches



*   Per level
    *   Size
    *   Type (I, D, I/D)
    *   Replacement policy
    *   Mapping (direct, associativity) 
    *   Line size
    *   Prefetching
    *   Miss cost modeling
    *   etc

Synchronization policies

Virtual Memory

DMA system descriptions

Multi-Processor System Topology


### **Appendix C: RISC-V Generated Architecture Description**

This is a complete, automatically generated machine description for RISC-V using our tool to scrape information from tablegen files.  We can automatically generate MDL specifications for all targets that have schedules and/or itineraries.  We include RISC-V here for illustrative purposes.

The “Schedule” td files for RISC-V are approximately 3231 lines of tablegen, describing three full schedule models and one “default” model.  The generated MDL file is ~374 lines of MDL.


```
//---------------------------------------------------------------------
// This file is autogenerated from an LLVM Target Description File.
//---------------------------------------------------------------------
import "RISCV_instructions.mdl"

//---------------------------------------------------------------------
// Pipeline phase definitions
//---------------------------------------------------------------------
protected phases RISCV { F1, E[1..1921] };

//---------------------------------------------------------------------
// CPU Description Classes (4 entries)
//---------------------------------------------------------------------
cpu RISCV("generic", "generic-rv32", "generic-rv64", "sifive-p450", "veyron-v1", "xiangshan-nanhu") {
}

cpu Rocket("rocket", "rocket-rv32", "rocket-rv64", "sifive-e20", "sifive-e21", "sifive-e24", "sifive-e31", "sifive-e34", "sifive-s21", "sifive-s51", "sifive-s54", "sifive-u54") {
  protected phases defaults { LOAD_PHASE=3 };
  issue(F1) s0;
  func_unit RocketUnitALU<0> U0();
  func_unit RocketUnitB<0> U1();
  func_unit RocketUnitFPALU<0> U2();
  func_unit RocketUnitFPDivSqrt<1> U3();
  func_unit RocketUnitIDiv<1> U4();
  func_unit RocketUnitIMul<0> U5();
  func_unit RocketUnitMem<0> U6();
}

cpu SiFive7("sifive-7-series", "sifive-e76", "sifive-s76", "sifive-u74", "sifive-x280") {
  protected phases defaults { LOAD_PHASE=3 };
  issue(F1) s0, s1;
  func_unit SiFive7FDiv<0> U0();
  func_unit SiFive7IDiv<0> U1();
  func_unit SiFive7PipeA<0> U2();
  func_unit SiFive7PipeB<0> U3();
  func_unit SiFive7VA<0> U4();
  func_unit SiFive7VCQ<0> U5();
  func_unit SiFive7VL<0> U6();
  func_unit SiFive7VS<0> U7();
}

cpu SyntacoreSCR1("syntacore-scr1-base", "syntacore-scr1-max") {
  protected phases defaults { LOAD_PHASE=2 };
  issue(F1) s0;
  func_unit SCR1_ALU<0> U0();
  func_unit SCR1_CFU<0> U1();
  func_unit SCR1_DIV<0> U2();
  func_unit SCR1_LSU<0> U3();
  func_unit SCR1_MUL<0> U4();
}

//---------------------------------------------------------------------
// Functional Unit Groups
//---------------------------------------------------------------------
func_group SiFive7PipeAB: SiFive7PipeA, SiFive7PipeB;

//---------------------------------------------------------------------
// Functional Unit Templates
//---------------------------------------------------------------------

//---------------------------------------------------------------------
// Subunit Definitions (232 entries)
//---------------------------------------------------------------------
subunit sub72() {{ Rocket : { def(E1, $X1); fus(RocketUnitALU, 1); fus(RocketUnitB, 1);} }}
subunit sub35() {{ Rocket : { def(E1, $X1); fus(RocketUnitB, 1);} }}
subunit sub78() {{ Rocket : { def(E1, $dst); fus(1); fus(Rocket, 0);} }}
subunit sub16() {{ Rocket : { def(E1, $dst); fus(RocketUnitALU, 1);} }}
subunit sub8() {{ Rocket : { def(E1, $rd); fus(1); fus(Rocket, 0);} }}
subunit sub75() {{ Rocket : { def(E1, $rd); fus(RocketUnitALU, 1); fus(RocketUnitB, 1);} }}
subunit sub0() {{ Rocket : { def(E1, $rd); fus(RocketUnitALU, 1);} }}
subunit sub68() {{ Rocket : { def(E1, $rd); fus(RocketUnitB, 1);} }}
subunit sub185() {{ Rocket : { def(E1, $rd); fus(RocketUnitMem, 1);} }}
subunit sub24() {{ Rocket : { def(E1, $rd_wb); fus(RocketUnitALU, 1);} }}
subunit sub44() {{ Rocket : { def(E1, $rs1); fus(RocketUnitALU, 1);} }}
subunit sub190() {{ Rocket : { def(E1, $rs1_up); fus(RocketUnitMem, 1);} }}
subunit sub21() {{ Rocket : { def(E1, $rs1_wb); fus(RocketUnitALU, 1);} }}
subunit sub196() {{ Rocket : { def(E1, $vd); fus(1); fus(Rocket, 0);} }}
subunit sub204() {{ Rocket : { def(E1, $vd_wb); fus(1); fus(Rocket, 0);} }}
subunit sub60() {{ Rocket : { def(E2, $rd); fus(RocketUnitFPALU, 1);} }}
subunit sub187() {{ Rocket : { def(E2, $rd); fus(RocketUnitMem, 1); def(E2, $rs2); fus(RocketUnitMem, 1);} }}
subunit sub4() {{ Rocket : { def(E2, $rd); fus(RocketUnitMem, 1);} }}
subunit sub61() {{ Rocket : { def(E20, $rd); fus(RocketUnitFPDivSqrt<20>, 1);} }}
subunit sub67() {{ Rocket : { def(E25, $rd); fus(RocketUnitFPDivSqrt<25>, 1);} }}
subunit sub39() {{ Rocket : { def(E3, $rd); fus(RocketUnitMem, 1);} }}
subunit sub51() {{ Rocket : { def(E33, $rd); fus(RocketUnitIDiv<33>, 1);} }}
subunit sub54() {{ Rocket : { def(E34, $rd); fus(RocketUnitIDiv<34>, 1);} }}
subunit sub59() {{ Rocket : { def(E4, $rd); fus(RocketUnitFPALU, 1);} }}
subunit sub70() {{ Rocket : { def(E4, $rd); fus(RocketUnitIMul, 1);} }}
subunit sub41() {{ Rocket : { def(E4, $rd_wb); fus(RocketUnitIMul, 1);} }}
subunit sub66() {{ Rocket : { def(E5, $rd); fus(RocketUnitFPALU, 1);} }}
subunit sub56() {{ Rocket : { def(E6, $rd); fus(RocketUnitFPALU, 1);} }}
subunit sub65() {{ Rocket : { def(E7, $rd); fus(RocketUnitFPALU, 1);} }}
subunit sub47() {{ Rocket : { fus(1); fus(Rocket, 0);} }}
subunit sub81() {{ Rocket : { fus(RocketUnitALU, 1); fus(RocketUnitB, 1);} }}
subunit sub12() {{ Rocket : { fus(RocketUnitB, 1);} }}
subunit sub193() {{ Rocket : { fus(RocketUnitMem, 1); fus(RocketUnitMem, 1);} }}
subunit sub31() {{ Rocket : { fus(RocketUnitMem, 1);} }}
subunit sub156() {{ SiFive7 : {
   if VLDSX0Pred { def(E4, $rd); fus(SiFive7VCQ&SiFive7VL<17>, 1); }
   else { def(E131, $rd); fus(SiFive7VCQ&SiFive7VL<129>, 1); }
} }}
subunit sub150() {{ SiFive7 : {
   if VLDSX0Pred { def(E4, $rd); fus(SiFive7VCQ&SiFive7VL<17>, 1); }
   else { def(E259, $rd); fus(SiFive7VCQ&SiFive7VL<257>, 1); }
} }}
subunit sub164() {{ SiFive7 : {
   if VLDSX0Pred { def(E4, $rd); fus(SiFive7VCQ&SiFive7VL<17>, 1); }
   else { def(E515, $rd); fus(SiFive7VCQ&SiFive7VL<513>, 1); }
} }}
subunit sub160() {{ SiFive7 : {
   if VLDSX0Pred { def(E4, $rd); fus(SiFive7VCQ&SiFive7VL<17>, 1); }
   else { def(E67, $rd); fus(SiFive7VCQ&SiFive7VL<65>, 1); }
} }}
subunit sub152() {{ SiFive7 : {
   if VLDSX0Pred { def(E4, $rd); fus(SiFive7VCQ&SiFive7VL<2>, 1); }
   else { def(E11, $rd); fus(SiFive7VCQ&SiFive7VL<9>, 1); }
} }}
subunit sub151() {{ SiFive7 : {
   if VLDSX0Pred { def(E4, $rd); fus(SiFive7VCQ&SiFive7VL<2>, 1); }
   else { def(E19, $rd); fus(SiFive7VCQ&SiFive7VL<17>, 1); }
} }}
subunit sub165() {{ SiFive7 : {
   if VLDSX0Pred { def(E4, $rd); fus(SiFive7VCQ&SiFive7VL<2>, 1); }
   else { def(E35, $rd); fus(SiFive7VCQ&SiFive7VL<33>, 1); }
} }}
subunit sub157() {{ SiFive7 : {
   if VLDSX0Pred { def(E4, $rd); fus(SiFive7VCQ&SiFive7VL<3>, 1); }
   else { def(E11, $rd); fus(SiFive7VCQ&SiFive7VL<9>, 1); }
} }}
subunit sub153() {{ SiFive7 : {
   if VLDSX0Pred { def(E4, $rd); fus(SiFive7VCQ&SiFive7VL<3>, 1); }
   else { def(E19, $rd); fus(SiFive7VCQ&SiFive7VL<17>, 1); }
} }}
subunit sub147() {{ SiFive7 : {
   if VLDSX0Pred { def(E4, $rd); fus(SiFive7VCQ&SiFive7VL<3>, 1); }
   else { def(E35, $rd); fus(SiFive7VCQ&SiFive7VL<33>, 1); }
} }}
subunit sub161() {{ SiFive7 : {
   if VLDSX0Pred { def(E4, $rd); fus(SiFive7VCQ&SiFive7VL<3>, 1); }
   else { def(E67, $rd); fus(SiFive7VCQ&SiFive7VL<65>, 1); }
} }}
subunit sub162() {{ SiFive7 : {
   if VLDSX0Pred { def(E4, $rd); fus(SiFive7VCQ&SiFive7VL<5>, 1); }
   else { def(E131, $rd); fus(SiFive7VCQ&SiFive7VL<129>, 1); }
} }}
subunit sub158() {{ SiFive7 : {
   if VLDSX0Pred { def(E4, $rd); fus(SiFive7VCQ&SiFive7VL<5>, 1); }
   else { def(E19, $rd); fus(SiFive7VCQ&SiFive7VL<17>, 1); }
} }}
subunit sub154() {{ SiFive7 : {
   if VLDSX0Pred { def(E4, $rd); fus(SiFive7VCQ&SiFive7VL<5>, 1); }
   else { def(E35, $rd); fus(SiFive7VCQ&SiFive7VL<33>, 1); }
} }}
subunit sub148() {{ SiFive7 : {
   if VLDSX0Pred { def(E4, $rd); fus(SiFive7VCQ&SiFive7VL<5>, 1); }
   else { def(E67, $rd); fus(SiFive7VCQ&SiFive7VL<65>, 1); }
} }}
subunit sub149() {{ SiFive7 : {
   if VLDSX0Pred { def(E4, $rd); fus(SiFive7VCQ&SiFive7VL<9>, 1); }
   else { def(E131, $rd); fus(SiFive7VCQ&SiFive7VL<129>, 1); }
} }}
subunit sub163() {{ SiFive7 : {
   if VLDSX0Pred { def(E4, $rd); fus(SiFive7VCQ&SiFive7VL<9>, 1); }
   else { def(E259, $rd); fus(SiFive7VCQ&SiFive7VL<257>, 1); }
} }}
subunit sub159() {{ SiFive7 : {
   if VLDSX0Pred { def(E4, $rd); fus(SiFive7VCQ&SiFive7VL<9>, 1); }
   else { def(E35, $rd); fus(SiFive7VCQ&SiFive7VL<33>, 1); }
} }}
subunit sub155() {{ SiFive7 : {
   if VLDSX0Pred { def(E4, $rd); fus(SiFive7VCQ&SiFive7VL<9>, 1); }
   else { def(E67, $rd); fus(SiFive7VCQ&SiFive7VL<65>, 1); }
} }}
subunit sub222() {{ SiFive7 : {
   if VLDSX0Pred { def(E4, $vd); fus(SiFive7VCQ&SiFive7VL<17>, 1); }
   else { def(E131, $vd); fus(SiFive7VCQ&SiFive7VL<129>, 1); }
} }}
subunit sub221() {{ SiFive7 : {
   if VLDSX0Pred { def(E4, $vd); fus(SiFive7VCQ&SiFive7VL<17>, 1); }
   else { def(E259, $vd); fus(SiFive7VCQ&SiFive7VL<257>, 1); }
} }}
subunit sub224() {{ SiFive7 : {
   if VLDSX0Pred { def(E4, $vd); fus(SiFive7VCQ&SiFive7VL<17>, 1); }
   else { def(E515, $vd); fus(SiFive7VCQ&SiFive7VL<513>, 1); }
} }}
subunit sub223() {{ SiFive7 : {
   if VLDSX0Pred { def(E4, $vd); fus(SiFive7VCQ&SiFive7VL<17>, 1); }
   else { def(E67, $vd); fus(SiFive7VCQ&SiFive7VL<65>, 1); }
} }}
subunit sub15() {{ SiFive7 : { def(E1, $rd); fus(1); fus(SiFive7, 0);} }}
subunit sub186() {{ SiFive7 : { def(E1, $rd); fus(SiFive7PipeA, 1);} }}
subunit sub20() {{ SiFive7 : { def(E1, $rd); fus(SiFive7PipeB, 1);} }}
subunit sub191() {{ SiFive7 : { def(E1, $rs1_up); fus(SiFive7PipeA, 1);} }}
subunit sub96() {{ SiFive7 : { def(E11, $rd); fus(SiFive7VCQ&SiFive7VA<9>, 1);} }}
subunit sub145() {{ SiFive7 : { def(E11, $rd); fus(SiFive7VCQ&SiFive7VL<9>, 1);} }}
subunit sub102() {{ SiFive7 : { def(E112, $rd); fus(SiFive7VCQ&SiFive7VA<113>, 1);} }}
subunit sub103() {{ SiFive7 : { def(E114, $rd); fus(SiFive7VCQ&SiFive7VA<115>, 1);} }}
subunit sub101() {{ SiFive7 : { def(E120, $rd); fus(SiFive7VCQ&SiFive7VA<121>, 1);} }}
subunit sub98() {{ SiFive7 : { def(E131, $rd); fus(SiFive7VCQ&SiFive7VA<129>, 1);} }}
subunit sub142() {{ SiFive7 : { def(E131, $rd); fus(SiFive7VCQ&SiFive7VL<129>, 1);} }}
subunit sub218() {{ SiFive7 : { def(E131, $vd); fus(SiFive7VCQ&SiFive7VL<129>, 1);} }}
subunit sub63() {{ SiFive7 : { def(E14, $rd); fus(SiFive7PipeB&SiFive7FDiv<13>, 1);} }}
subunit sub134() {{ SiFive7 : { def(E1536, $rd); fus(SiFive7VCQ&SiFive7VA<1537>, 1);} }}
subunit sub210() {{ SiFive7 : { def(E1536, $vd); fus(SiFive7VCQ&SiFive7VA<1537>, 1);} }}
subunit sub95() {{ SiFive7 : { def(E19, $rd); fus(SiFive7VCQ&SiFive7VA<17>, 1);} }}
subunit sub144() {{ SiFive7 : { def(E19, $rd); fus(SiFive7VCQ&SiFive7VL<17>, 1);} }}
subunit sub227() {{ SiFive7 : { def(E19, $vd); fus(SiFive7VCQ&SiFive7VA<17>, 1);} }}
subunit sub130() {{ SiFive7 : { def(E192, $rd); fus(SiFive7VCQ&SiFive7VA<193>, 1);} }}
subunit sub113() {{ SiFive7 : { def(E1920, $rd); fus(SiFive7VCQ&SiFive7VA<1921>, 1);} }}
subunit sub202() {{ SiFive7 : { def(E1920, $vd); fus(SiFive7VCQ&SiFive7VA<1921>, 1);} }}
subunit sub30() {{ SiFive7 : { def(E2, $rd); fus(SiFive7PipeA, 1);} }}
subunit sub105() {{ SiFive7 : { def(E224, $rd); fus(SiFive7VCQ&SiFive7VA<225>, 1);} }}
subunit sub106() {{ SiFive7 : { def(E228, $rd); fus(SiFive7VCQ&SiFive7VA<229>, 1);} }}
subunit sub104() {{ SiFive7 : { def(E240, $rd); fus(SiFive7VCQ&SiFive7VA<241>, 1);} }}
subunit sub99() {{ SiFive7 : { def(E259, $rd); fus(SiFive7VCQ&SiFive7VA<257>, 1);} }}
subunit sub143() {{ SiFive7 : { def(E259, $rd); fus(SiFive7VCQ&SiFive7VL<257>, 1);} }}
subunit sub217() {{ SiFive7 : { def(E259, $vd); fus(SiFive7VCQ&SiFive7VL<257>, 1);} }}
subunit sub64() {{ SiFive7 : { def(E27, $rd); fus(SiFive7PipeB&SiFive7FDiv<26>, 1);} }}
subunit sub73() {{ SiFive7 : { def(E3, $X1); fus(SiFive7PipeAB, 1); fus(SiFive7PipeB, 1);} }}
subunit sub38() {{ SiFive7 : { def(E3, $X1); fus(SiFive7PipeB, 1); use(E3, $rs1);} }}
subunit sub36() {{ SiFive7 : { def(E3, $X1); fus(SiFive7PipeB, 1);} }}
subunit sub79() {{ SiFive7 : { def(E3, $dst); fus(SiFive7PipeA&SiFive7PipeB, 2);} }}
subunit sub17() {{ SiFive7 : { def(E3, $dst); fus(SiFive7PipeAB, 1);} }}
subunit sub188() {{ SiFive7 : { def(E3, $rd); fus(SiFive7PipeA, 1); def(E3, $rs2); fus(SiFive7PipeA, 1);} }}
subunit sub231() {{ SiFive7 : { def(E3, $rd); fus(SiFive7PipeA, 1); use(E3, $rs1); use(E3, $rs2);} }}
subunit sub179() {{ SiFive7 : { def(E3, $rd); fus(SiFive7PipeA, 1); use(E3, $rs1);} }}
subunit sub5() {{ SiFive7 : { def(E3, $rd); fus(SiFive7PipeA, 1);} }}
subunit sub76() {{ SiFive7 : { def(E3, $rd); fus(SiFive7PipeAB, 1); fus(SiFive7PipeB, 1);} }}
subunit sub1() {{ SiFive7 : { def(E3, $rd); fus(SiFive7PipeAB, 1); use(E3, $rs1); use(E3, $rs2);} }}
subunit sub3() {{ SiFive7 : { def(E3, $rd); fus(SiFive7PipeAB, 1); use(E3, $rs1);} }}
subunit sub7() {{ SiFive7 : { def(E3, $rd); fus(SiFive7PipeAB, 1);} }}
subunit sub9() {{ SiFive7 : { def(E3, $rd); fus(SiFive7PipeB, 1); use(E3, $rs1); use(E3, $rs2);} }}
subunit sub11() {{ SiFive7 : { def(E3, $rd); fus(SiFive7PipeB, 1); use(E3, $rs1);} }}
subunit sub19() {{ SiFive7 : { def(E3, $rd); fus(SiFive7PipeB, 1);} }}
subunit sub27() {{ SiFive7 : { def(E3, $rd_wb); fus(SiFive7PipeAB, 1); use(E3, $rd); use(E3, $rs2);} }}
subunit sub25() {{ SiFive7 : { def(E3, $rd_wb); fus(SiFive7PipeAB, 1); use(E3, $rd);} }}
subunit sub42() {{ SiFive7 : { def(E3, $rd_wb); fus(SiFive7PipeB, 1);} }}
subunit sub45() {{ SiFive7 : { def(E3, $rs1); fus(SiFive7PipeAB, 1); use(E3, $rs2);} }}
subunit sub22() {{ SiFive7 : { def(E3, $rs1_wb); fus(SiFive7PipeAB, 1); use(E3, $rs1); use(E3, $rs2);} }}
subunit sub28() {{ SiFive7 : { def(E3, $rs1_wb); fus(SiFive7PipeAB, 1); use(E3, $rs1);} }}
subunit sub116() {{ SiFive7 : { def(E30, $rd); fus(SiFive7VCQ&SiFive7VA<31>, 1);} }}
subunit sub119() {{ SiFive7 : { def(E32, $rd); fus(SiFive7VCQ&SiFive7VA<33>, 1);} }}
subunit sub55() {{ SiFive7 : { def(E34, $rd); fus(SiFive7PipeB&SiFive7IDiv<33>, 1);} }}
subunit sub122() {{ SiFive7 : { def(E34, $rd); fus(SiFive7VCQ&SiFive7VA<35>, 1);} }}
subunit sub94() {{ SiFive7 : { def(E35, $rd); fus(SiFive7VCQ&SiFive7VA<33>, 1);} }}
subunit sub140() {{ SiFive7 : { def(E35, $rd); fus(SiFive7VCQ&SiFive7VL<33>, 1);} }}
subunit sub225() {{ SiFive7 : { def(E35, $vd); fus(SiFive7VCQ&SiFive7VL<33>, 1);} }}
subunit sub129() {{ SiFive7 : { def(E36, $rd); fus(SiFive7VCQ&SiFive7VA<37>, 1);} }}
subunit sub118() {{ SiFive7 : { def(E37, $rd); fus(SiFive7VCQ&SiFive7VA<38>, 1);} }}
subunit sub125() {{ SiFive7 : { def(E38, $rd); fus(SiFive7VCQ&SiFive7VA<39>, 1);} }}
subunit sub132() {{ SiFive7 : { def(E384, $rd); fus(SiFive7VCQ&SiFive7VA<385>, 1);} }}
subunit sub121() {{ SiFive7 : { def(E39, $rd); fus(SiFive7VCQ&SiFive7VA<40>, 1);} }}
subunit sub92() {{ SiFive7 : { def(E4, $rd); fus(SiFive7VCQ&SiFive7VA<17>, 1);} }}
subunit sub93() {{ SiFive7 : { def(E4, $rd); fus(SiFive7VCQ&SiFive7VA<2>, 1);} }}
subunit sub89() {{ SiFive7 : { def(E4, $rd); fus(SiFive7VCQ&SiFive7VA<3>, 1);} }}
subunit sub90() {{ SiFive7 : { def(E4, $rd); fus(SiFive7VCQ&SiFive7VA<5>, 1);} }}
subunit sub91() {{ SiFive7 : { def(E4, $rd); fus(SiFive7VCQ&SiFive7VA<9>, 1);} }}
subunit sub138() {{ SiFive7 : { def(E4, $rd); fus(SiFive7VCQ&SiFive7VL<17>, 1);} }}
subunit sub139() {{ SiFive7 : { def(E4, $rd); fus(SiFive7VCQ&SiFive7VL<2>, 1);} }}
subunit sub135() {{ SiFive7 : { def(E4, $rd); fus(SiFive7VCQ&SiFive7VL<3>, 1);} }}
subunit sub136() {{ SiFive7 : { def(E4, $rd); fus(SiFive7VCQ&SiFive7VL<5>, 1);} }}
subunit sub137() {{ SiFive7 : { def(E4, $rd); fus(SiFive7VCQ&SiFive7VL<9>, 1);} }}
subunit sub199() {{ SiFive7 : { def(E4, $vd); fus(SiFive7VCQ&SiFive7VA<17>, 1);} }}
subunit sub207() {{ SiFive7 : { def(E4, $vd); fus(SiFive7VCQ&SiFive7VA<2>, 1);} }}
subunit sub201() {{ SiFive7 : { def(E4, $vd); fus(SiFive7VCQ&SiFive7VA<3>, 1);} }}
subunit sub228() {{ SiFive7 : { def(E4, $vd); fus(SiFive7VCQ&SiFive7VA<5>, 1);} }}
subunit sub229() {{ SiFive7 : { def(E4, $vd); fus(SiFive7VCQ&SiFive7VA<9>, 1);} }}
subunit sub216() {{ SiFive7 : { def(E4, $vd); fus(SiFive7VCQ&SiFive7VL<17>, 1);} }}
subunit sub213() {{ SiFive7 : { def(E4, $vd); fus(SiFive7VCQ&SiFive7VL<3>, 1);} }}
subunit sub214() {{ SiFive7 : { def(E4, $vd); fus(SiFive7VCQ&SiFive7VL<5>, 1);} }}
subunit sub215() {{ SiFive7 : { def(E4, $vd); fus(SiFive7VCQ&SiFive7VL<9>, 1);} }}
subunit sub226() {{ SiFive7 : { def(E4, $vd_wb); fus(SiFive7VCQ&SiFive7VA<17>, 1);} }}
subunit sub208() {{ SiFive7 : { def(E4, $vd_wb); fus(SiFive7VCQ&SiFive7VA<2>, 1);} }}
subunit sub128() {{ SiFive7 : { def(E41, $rd); fus(SiFive7VCQ&SiFive7VA<42>, 1);} }}
subunit sub117() {{ SiFive7 : { def(E42, $rd); fus(SiFive7VCQ&SiFive7VA<43>, 1);} }}
subunit sub124() {{ SiFive7 : { def(E43, $rd); fus(SiFive7VCQ&SiFive7VA<44>, 1);} }}
subunit sub120() {{ SiFive7 : { def(E44, $rd); fus(SiFive7VCQ&SiFive7VA<45>, 1);} }}
subunit sub108() {{ SiFive7 : { def(E448, $rd); fus(SiFive7VCQ&SiFive7VA<449>, 1);} }}
subunit sub109() {{ SiFive7 : { def(E456, $rd); fus(SiFive7VCQ&SiFive7VA<457>, 1);} }}
subunit sub127() {{ SiFive7 : { def(E46, $rd); fus(SiFive7VCQ&SiFive7VA<47>, 1);} }}
subunit sub170() {{ SiFive7 : { def(E47, $rd); fus(SiFive7VCQ&SiFive7VA<48>, 1);} }}
subunit sub123() {{ SiFive7 : { def(E48, $rd); fus(SiFive7VCQ&SiFive7VA<49>, 1);} }}
subunit sub107() {{ SiFive7 : { def(E480, $rd); fus(SiFive7VCQ&SiFive7VA<481>, 1);} }}
subunit sub171() {{ SiFive7 : { def(E49, $rd); fus(SiFive7VCQ&SiFive7VA<50>, 1);} }}
subunit sub58() {{ SiFive7 : { def(E5, $rd); fus(SiFive7PipeB, 1);} }}
subunit sub168() {{ SiFive7 : { def(E5, $rd); fus(SiFive7VCQ&SiFive7VA<3>, 1);} }}
subunit sub167() {{ SiFive7 : { def(E5, $rd); fus(SiFive7VCQ&SiFive7VL<3>, 1);} }}
subunit sub126() {{ SiFive7 : { def(E51, $rd); fus(SiFive7VCQ&SiFive7VA<52>, 1);} }}
subunit sub100() {{ SiFive7 : { def(E515, $rd); fus(SiFive7VCQ&SiFive7VA<513>, 1);} }}
subunit sub146() {{ SiFive7 : { def(E515, $rd); fus(SiFive7VCQ&SiFive7VL<513>, 1);} }}
subunit sub200() {{ SiFive7 : { def(E515, $vd); fus(SiFive7VCQ&SiFive7VA<513>, 1);} }}
subunit sub220() {{ SiFive7 : { def(E515, $vd); fus(SiFive7VCQ&SiFive7VL<513>, 1);} }}
subunit sub172() {{ SiFive7 : { def(E53, $rd); fus(SiFive7VCQ&SiFive7VA<54>, 1);} }}
subunit sub62() {{ SiFive7 : { def(E56, $rd); fus(SiFive7PipeB&SiFive7FDiv<55>, 1);} }}
subunit sub115() {{ SiFive7 : { def(E56, $rd); fus(SiFive7VCQ&SiFive7VA<57>, 1);} }}
subunit sub209() {{ SiFive7 : { def(E56, $vd); fus(SiFive7VCQ&SiFive7VA<57>, 1);} }}
subunit sub114() {{ SiFive7 : { def(E60, $rd); fus(SiFive7VCQ&SiFive7VA<61>, 1);} }}
subunit sub173() {{ SiFive7 : { def(E61, $rd); fus(SiFive7VCQ&SiFive7VA<62>, 1);} }}
subunit sub230() {{ SiFive7 : { def(E61, $vd); fus(SiFive7VCQ&SiFive7VA<62>, 1);} }}
subunit sub52() {{ SiFive7 : { def(E66, $rd); fus(SiFive7PipeB&SiFive7IDiv<65>, 1);} }}
subunit sub97() {{ SiFive7 : { def(E67, $rd); fus(SiFive7VCQ&SiFive7VA<65>, 1);} }}
subunit sub141() {{ SiFive7 : { def(E67, $rd); fus(SiFive7VCQ&SiFive7VL<65>, 1);} }}
subunit sub219() {{ SiFive7 : { def(E67, $vd); fus(SiFive7VCQ&SiFive7VL<65>, 1);} }}
subunit sub57() {{ SiFive7 : { def(E7, $rd); fus(SiFive7PipeB, 1);} }}
subunit sub169() {{ SiFive7 : { def(E7, $rd); fus(SiFive7VCQ&SiFive7VA<5>, 1);} }}
subunit sub166() {{ SiFive7 : { def(E7, $rd); fus(SiFive7VCQ&SiFive7VL<5>, 1);} }}
subunit sub133() {{ SiFive7 : { def(E768, $rd); fus(SiFive7VCQ&SiFive7VA<769>, 1);} }}
subunit sub87() {{ SiFive7 : { def(E8, $rd); fus(SiFive7VCQ&SiFive7VA<17>, 1);} }}
subunit sub88() {{ SiFive7 : { def(E8, $rd); fus(SiFive7VCQ&SiFive7VA<2>, 1);} }}
subunit sub84() {{ SiFive7 : { def(E8, $rd); fus(SiFive7VCQ&SiFive7VA<3>, 1);} }}
subunit sub85() {{ SiFive7 : { def(E8, $rd); fus(SiFive7VCQ&SiFive7VA<5>, 1);} }}
subunit sub86() {{ SiFive7 : { def(E8, $rd); fus(SiFive7VCQ&SiFive7VA<9>, 1);} }}
subunit sub197() {{ SiFive7 : { def(E8, $vd); fus(SiFive7VCQ&SiFive7VA<17>, 1);} }}
subunit sub211() {{ SiFive7 : { def(E8, $vd); fus(SiFive7VCQ&SiFive7VA<9>, 1);} }}
subunit sub205() {{ SiFive7 : { def(E8, $vd_wb); fus(SiFive7VCQ&SiFive7VA<17>, 1);} }}
subunit sub212() {{ SiFive7 : { def(E8, $vd_wb); fus(SiFive7VCQ&SiFive7VA<9>, 1);} }}
subunit sub111() {{ SiFive7 : { def(E896, $rd); fus(SiFive7VCQ&SiFive7VA<897>, 1);} }}
subunit sub112() {{ SiFive7 : { def(E912, $rd); fus(SiFive7VCQ&SiFive7VA<913>, 1);} }}
subunit sub131() {{ SiFive7 : { def(E96, $rd); fus(SiFive7VCQ&SiFive7VA<97>, 1);} }}
subunit sub110() {{ SiFive7 : { def(E960, $rd); fus(SiFive7VCQ&SiFive7VA<961>, 1);} }}
subunit sub203() {{ SiFive7 : { def(E960, $vd); fus(SiFive7VCQ&SiFive7VA<961>, 1);} }}
subunit sub48() {{ SiFive7 : { fus(1); fus(SiFive7, 0);} }}
subunit sub194() {{ SiFive7 : { fus(SiFive7PipeA, 1); fus(SiFive7PipeA, 1);} }}
subunit sub32() {{ SiFive7 : { fus(SiFive7PipeA, 1);} }}
subunit sub82() {{ SiFive7 : { fus(SiFive7PipeAB, 1); fus(SiFive7PipeB, 1); use(E3, $X2);} }}
subunit sub13() {{ SiFive7 : { fus(SiFive7PipeB, 1); use(E3, $rs1); use(E3, $rs2);} }}
subunit sub29() {{ SiFive7 : { fus(SiFive7PipeB, 1); use(E3, $rs1);} }}
subunit sub34() {{ SiFive7 : { fus(SiFive7PipeB, 1);} }}
subunit sub182() {{ SiFive7 : { fus(SiFive7VCQ&SiFive7VS<129>, 1);} }}
subunit sub177() {{ SiFive7 : { fus(SiFive7VCQ&SiFive7VS<17>, 1);} }}
subunit sub183() {{ SiFive7 : { fus(SiFive7VCQ&SiFive7VS<257>, 1);} }}
subunit sub178() {{ SiFive7 : { fus(SiFive7VCQ&SiFive7VS<2>, 1);} }}
subunit sub180() {{ SiFive7 : { fus(SiFive7VCQ&SiFive7VS<33>, 1);} }}
subunit sub174() {{ SiFive7 : { fus(SiFive7VCQ&SiFive7VS<3>, 1);} }}
subunit sub184() {{ SiFive7 : { fus(SiFive7VCQ&SiFive7VS<513>, 1);} }}
subunit sub175() {{ SiFive7 : { fus(SiFive7VCQ&SiFive7VS<5>, 1);} }}
subunit sub181() {{ SiFive7 : { fus(SiFive7VCQ&SiFive7VS<65>, 1);} }}
subunit sub176() {{ SiFive7 : { fus(SiFive7VCQ&SiFive7VS<9>, 1);} }}
subunit sub74() {{ SyntacoreSCR1 : { def(E1, $X1); fus(SCR1_ALU, 1); fus(SCR1_CFU, 1);} }}
subunit sub37() {{ SyntacoreSCR1 : { def(E1, $X1); fus(SCR1_CFU, 1);} }}
subunit sub80() {{ SyntacoreSCR1 : { def(E1, $dst); fus(1); fus(SyntacoreSCR1, 0);} }}
subunit sub18() {{ SyntacoreSCR1 : { def(E1, $dst); fus(SCR1_ALU, 1);} }}
subunit sub10() {{ SyntacoreSCR1 : { def(E1, $rd); fus(1); fus(SyntacoreSCR1, 0);} }}
subunit sub77() {{ SyntacoreSCR1 : { def(E1, $rd); fus(SCR1_ALU, 1); fus(SCR1_CFU, 1);} }}
subunit sub2() {{ SyntacoreSCR1 : { def(E1, $rd); fus(SCR1_ALU, 1);} }}
subunit sub69() {{ SyntacoreSCR1 : { def(E1, $rd); fus(SCR1_CFU, 1);} }}
subunit sub6() {{ SyntacoreSCR1 : { def(E1, $rd); fus(SCR1_LSU, 1);} }}
subunit sub71() {{ SyntacoreSCR1 : { def(E1, $rd); fus(SCR1_MUL, 1);} }}
subunit sub26() {{ SyntacoreSCR1 : { def(E1, $rd_wb); fus(SCR1_ALU, 1);} }}
subunit sub43() {{ SyntacoreSCR1 : { def(E1, $rd_wb); fus(SCR1_MUL, 1);} }}
subunit sub46() {{ SyntacoreSCR1 : { def(E1, $rs1); fus(SCR1_ALU, 1);} }}
subunit sub23() {{ SyntacoreSCR1 : { def(E1, $rs1_wb); fus(SCR1_ALU, 1);} }}
subunit sub198() {{ SyntacoreSCR1 : { def(E1, $vd); fus(1); fus(SyntacoreSCR1, 0);} }}
subunit sub206() {{ SyntacoreSCR1 : { def(E1, $vd_wb); fus(1); fus(SyntacoreSCR1, 0);} }}
subunit sub189() {{ SyntacoreSCR1 : { def(E2, $rd); fus(SCR1_LSU<2>, 1); def(E2, $rs2); fus(SCR1_LSU<2>, 1);} }}
subunit sub40() {{ SyntacoreSCR1 : { def(E2, $rd); fus(SCR1_LSU<2>, 1);} }}
subunit sub192() {{ SyntacoreSCR1 : { def(E2, $rs1_up); fus(SCR1_LSU<2>, 1);} }}
subunit sub53() {{ SyntacoreSCR1 : { def(E33, $rd); fus(SCR1_DIV<33>, 1);} }}
subunit sub49() {{ SyntacoreSCR1 : { fus(1); fus(SyntacoreSCR1, 0);} }}
subunit sub83() {{ SyntacoreSCR1 : { fus(SCR1_ALU, 1); fus(SCR1_CFU, 1);} }}
subunit sub14() {{ SyntacoreSCR1 : { fus(SCR1_CFU, 1);} }}
subunit sub33() {{ SyntacoreSCR1 : { fus(SCR1_LSU, 1);} }}
subunit sub195() {{ SyntacoreSCR1 : { fus(SCR1_LSU<2>, 1); fus(SCR1_LSU<2>, 1);} }}
subunit sub50() {{ SyntacoreSCR1 : { fus(SCR1_LSU<2>, 1);} }}

//---------------------------------------------------------------------
// Predicate Definitions
//---------------------------------------------------------------------

predicate VLDSX0Pred : CheckRegOperand<3,X0>;

