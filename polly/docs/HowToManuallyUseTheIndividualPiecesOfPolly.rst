==================================================
How to manually use the Individual pieces of Polly
==================================================

Execute the individual Polly passes manually
============================================

.. sectionauthor:: Singapuram Sanjay Srivallabh

This example presents the individual passes that are involved when optimizing
code with Polly. We show how to execute them individually and explain for
each which analysis is performed or what transformation is applied. In this
example the polyhedral transformation is user-provided to show how much
performance improvement can be expected by an optimal automatic optimizer.

1. **Create LLVM-IR from the C code**
-------------------------------------
        Polly works on LLVM-IR. Hence it is necessary to translate the source
        files into LLVM-IR. If more than one file should be optimized the
        files can be combined into a single file with llvm-link.

        .. code-block:: console

                clang -S -emit-llvm matmul.c -Xclang -disable-O0-optnone -o matmul.ll


2. **Prepare the LLVM-IR for Polly**
------------------------------------

        Polly is only able to work with code that matches a canonical form.
        To translate the LLVM-IR into this form we use a set of
        canonicalication passes. They are scheduled by using
        '-polly-canonicalize'.

        .. code-block:: console

                opt -S -polly-canonicalize matmul.ll -o matmul.preopt.ll

3. **Show the SCoPs detected by Polly (optional)**
--------------------------------------------------

        To understand if Polly was able to detect SCoPs, we print the structure
        of the detected SCoPs. In our example two SCoPs are detected. One in
        'init_array' the other in 'main'.

        .. code-block:: console

                $ opt -basic-aa -polly-ast -analyze matmul.preopt.ll -polly-process-unprofitable -polly-use-llvm-names

        .. code-block:: guess

                :: isl ast :: init_array :: %for.cond1.preheader---%for.end19

                if (1)

                    for (int c0 = 0; c0 <= 1535; c0 += 1)
                      for (int c1 = 0; c1 <= 1535; c1 += 1)
                        Stmt_for_body3(c0, c1);

                else
                    {  /* original code */ }

                :: isl ast :: main :: %for.cond1.preheader---%for.end30

                if (1)

                    for (int c0 = 0; c0 <= 1535; c0 += 1)
                      for (int c1 = 0; c1 <= 1535; c1 += 1) {
                        Stmt_for_body3(c0, c1);
                        for (int c2 = 0; c2 <= 1535; c2 += 1)
                          Stmt_for_body8(c0, c1, c2);
                      }

                else
                    {  /* original code */ }

4. **Highlight the detected SCoPs in the CFGs of the program (requires graphviz/dotty)**
----------------------------------------------------------------------------------------

        Polly can use graphviz to graphically show a CFG in which the detected
        SCoPs are highlighted. It can also create '.dot' files that can be
        translated by the 'dot' utility into various graphic formats.


        .. code-block:: console

                $ opt -polly-use-llvm-names -basic-aa -view-scops -disable-output matmul.preopt.ll
                $ opt -polly-use-llvm-names -basic-aa -view-scops-only -disable-output matmul.preopt.ll

        The output for the different functions:

        - view-scops : main_, init_array_, print_array_
        - view-scops-only : main-scopsonly_, init_array-scopsonly_, print_array-scopsonly_

.. _main:  http://polly.llvm.org/experiments/matmul/scops.main.dot.png
.. _init_array: http://polly.llvm.org/experiments/matmul/scops.init_array.dot.png
.. _print_array: http://polly.llvm.org/experiments/matmul/scops.print_array.dot.png
.. _main-scopsonly: http://polly.llvm.org/experiments/matmul/scopsonly.main.dot.png
.. _init_array-scopsonly: http://polly.llvm.org/experiments/matmul/scopsonly.init_array.dot.png
.. _print_array-scopsonly: http://polly.llvm.org/experiments/matmul/scopsonly.print_array.dot.png

5. **View the polyhedral representation of the SCoPs**
------------------------------------------------------

        .. code-block:: console

                $ opt -polly-use-llvm-names -basic-aa -polly-scops -analyze matmul.preopt.ll -polly-process-unprofitable

        .. code-block:: guess

                [...]Printing analysis 'Polly - Create polyhedral description of Scops' for region: 'for.cond1.preheader => for.end19' in function 'init_array':
                    Function: init_array
                    Region: %for.cond1.preheader---%for.end19
                    Max Loop Depth:  2
                        Invariant Accesses: {
                        }
                        Context:
                        {  :  }
                        Assumed Context:
                        {  :  }
                        Invalid Context:
                        {  : 1 = 0 }
                        Arrays {
                            float MemRef_A[*][1536]; // Element size 4
                            float MemRef_B[*][1536]; // Element size 4
                        }
                        Arrays (Bounds as pw_affs) {
                            float MemRef_A[*][ { [] -> [(1536)] } ]; // Element size 4
                            float MemRef_B[*][ { [] -> [(1536)] } ]; // Element size 4
                        }
                        Alias Groups (0):
                            n/a
                        Statements {
    	                    Stmt_for_body3
                                Domain :=
                                    { Stmt_for_body3[i0, i1] : 0 <= i0 <= 1535 and 0 <= i1 <= 1535 };
                                Schedule :=
                                    { Stmt_for_body3[i0, i1] -> [i0, i1] };
                                MustWriteAccess :=	[Reduction Type: NONE] [Scalar: 0]
                                    { Stmt_for_body3[i0, i1] -> MemRef_A[i0, i1] };
                                MustWriteAccess :=	[Reduction Type: NONE] [Scalar: 0]
                                    { Stmt_for_body3[i0, i1] -> MemRef_B[i0, i1] };
                        }
                [...]Printing analysis 'Polly - Create polyhedral description of Scops' for region: 'for.cond1.preheader => for.end30' in function 'main':
                    Function: main
                    Region: %for.cond1.preheader---%for.end30
                    Max Loop Depth:  3
                    Invariant Accesses: {
                    }
                    Context:
                    {  :  }
                    Assumed Context:
                    {  :  }
                    Invalid Context:
                    {  : 1 = 0 }
                    Arrays {
                        float MemRef_C[*][1536]; // Element size 4
                        float MemRef_A[*][1536]; // Element size 4
                        float MemRef_B[*][1536]; // Element size 4
                    }
                    Arrays (Bounds as pw_affs) {
                        float MemRef_C[*][ { [] -> [(1536)] } ]; // Element size 4
                        float MemRef_A[*][ { [] -> [(1536)] } ]; // Element size 4
                        float MemRef_B[*][ { [] -> [(1536)] } ]; // Element size 4
                    }
                    Alias Groups (0):
                        n/a
                    Statements {
                    	Stmt_for_body3
                            Domain :=
                                { Stmt_for_body3[i0, i1] : 0 <= i0 <= 1535 and 0 <= i1 <= 1535 };
                            Schedule :=
                                { Stmt_for_body3[i0, i1] -> [i0, i1, 0, 0] };
                            MustWriteAccess :=	[Reduction Type: NONE] [Scalar: 0]
                                { Stmt_for_body3[i0, i1] -> MemRef_C[i0, i1] };
                    	Stmt_for_body8
                            Domain :=
                                { Stmt_for_body8[i0, i1, i2] : 0 <= i0 <= 1535 and 0 <= i1 <= 1535 and 0 <= i2 <= 1535 };
                            Schedule :=
                                { Stmt_for_body8[i0, i1, i2] -> [i0, i1, 1, i2] };
                            ReadAccess :=	[Reduction Type: NONE] [Scalar: 0]
                                { Stmt_for_body8[i0, i1, i2] -> MemRef_C[i0, i1] };
                            ReadAccess :=	[Reduction Type: NONE] [Scalar: 0]
                                { Stmt_for_body8[i0, i1, i2] -> MemRef_A[i0, i2] };
                            ReadAccess :=	[Reduction Type: NONE] [Scalar: 0]
                                { Stmt_for_body8[i0, i1, i2] -> MemRef_B[i2, i1] };
                            MustWriteAccess :=	[Reduction Type: NONE] [Scalar: 0]
                                { Stmt_for_body8[i0, i1, i2] -> MemRef_C[i0, i1] };
                    }


6. **Show the dependences for the SCoPs**
-----------------------------------------

        .. code-block:: console

	        $ opt -basic-aa -polly-use-llvm-names -polly-dependences -analyze matmul.preopt.ll -polly-process-unprofitable

        .. code-block:: guess

        	[...]Printing analysis 'Polly - Calculate dependences' for region: 'for.cond1.preheader => for.end19' in function 'init_array':
        		RAW dependences:
        			{  }
        		WAR dependences:
        			{  }
        		WAW dependences:
        			{  }
        		Reduction dependences:
        			n/a
        		Transitive closure of reduction dependences:
        			{  }
        	[...]Printing analysis 'Polly - Calculate dependences' for region: 'for.cond1.preheader => for.end30' in function 'main':
        		RAW dependences:
        			{ Stmt_for_body3[i0, i1] -> Stmt_for_body8[i0, i1, 0] : 0 <= i0 <= 1535 and 0 <= i1 <= 1535; Stmt_for_body8[i0, i1, i2] -> Stmt_for_body8[i0, i1, 1 + i2] : 0 <= i0 <= 1535 and 0 <= i1 <= 1535 and 0 <= i2 <= 1534 }
        		WAR dependences:
        			{  }
        		WAW dependences:
        			{ Stmt_for_body3[i0, i1] -> Stmt_for_body8[i0, i1, 0] : 0 <= i0 <= 1535 and 0 <= i1 <= 1535; Stmt_for_body8[i0, i1, i2] -> Stmt_for_body8[i0, i1, 1 + i2] : 0 <= i0 <= 1535 and 0 <= i1 <= 1535 and 0 <= i2 <= 1534 }
        		Reduction dependences:
        			n/a
        		Transitive closure of reduction dependences:
        			{  }

7. **Export jscop files**
-------------------------

        .. code-block:: console

        	$ opt -basic-aa -polly-use-llvm-names -polly-export-jscop matmul.preopt.ll -polly-process-unprofitable

        .. code-block:: guess

	        [...]Writing JScop '%for.cond1.preheader---%for.end19' in function 'init_array' to './init_array___%for.cond1.preheader---%for.end19.jscop'.

	        Writing JScop '%for.cond1.preheader---%for.end30' in function 'main' to './main___%for.cond1.preheader---%for.end30.jscop'.



8. **Import the changed jscop files and print the updated SCoP structure (optional)**
-------------------------------------------------------------------------------------

	Polly can reimport jscop files, in which the schedules of the statements
        are changed. These changed schedules are used to descripe
        transformations. It is possible to import different jscop files by
        providing the postfix of the jscop file that is imported.

	We apply three different transformations on the SCoP in the main
        function. The jscop files describing these transformations are
        hand written (and available in docs/experiments/matmul).

	**No Polly**

	As a baseline we do not call any Polly code generation, but only apply the normal -O3 optimizations.

	.. code-block:: console

		$ opt -basic-aa -polly-use-llvm-names matmul.preopt.ll -polly-import-jscop -polly-ast -analyze -polly-process-unprofitable

	.. code-block:: c

		[...]
		:: isl ast :: main :: %for.cond1.preheader---%for.end30
		
		if (1)
		
		    for (int c0 = 0; c0 <= 1535; c0 += 1)
		      for (int c1 = 0; c1 <= 1535; c1 += 1) {
		        Stmt_for_body3(c0, c1);
		        for (int c3 = 0; c3 <= 1535; c3 += 1)
		          Stmt_for_body8(c0, c1, c3);
		      }
		
		else
		    {  /* original code */ }
		[...]

	**Loop Interchange (and Fission to allow the interchange)**

	We split the loops and can now apply an interchange of the loop dimensions that enumerate Stmt_for_body8.

	.. Although I feel (and have created a .jscop) we can avoid splitting the loops.

	.. code-block:: console

		$ opt -basic-aa -polly-use-llvm-names matmul.preopt.ll -polly-import-jscop -polly-import-jscop-postfix=interchanged -polly-ast -analyze -polly-process-unprofitable

	.. code-block:: c

		[...]
		:: isl ast :: main :: %for.cond1.preheader---%for.end30

		if (1)

		    {
		      for (int c1 = 0; c1 <= 1535; c1 += 1)
		        for (int c2 = 0; c2 <= 1535; c2 += 1)
		          Stmt_for_body3(c1, c2);
		      for (int c1 = 0; c1 <= 1535; c1 += 1)
		        for (int c2 = 0; c2 <= 1535; c2 += 1)
		          for (int c3 = 0; c3 <= 1535; c3 += 1)
		            Stmt_for_body8(c1, c3, c2);
		    }

		else
		    {  /* original code */ }
		[...]

	**Interchange + Tiling**

	In addition to the interchange we now tile the second loop nest.

	.. code-block:: console

		$ opt -basic-aa -polly-use-llvm-names matmul.preopt.ll -polly-import-jscop -polly-import-jscop-postfix=interchanged+tiled -polly-ast -analyze -polly-process-unprofitable

	.. code-block:: c

		[...]
		:: isl ast :: main :: %for.cond1.preheader---%for.end30

		if (1)

		    {
		      for (int c1 = 0; c1 <= 1535; c1 += 1)
		        for (int c2 = 0; c2 <= 1535; c2 += 1)
		          Stmt_for_body3(c1, c2);
		      for (int c1 = 0; c1 <= 1535; c1 += 64)
		        for (int c2 = 0; c2 <= 1535; c2 += 64)
		          for (int c3 = 0; c3 <= 1535; c3 += 64)
		            for (int c4 = c1; c4 <= c1 + 63; c4 += 1)
		              for (int c5 = c3; c5 <= c3 + 63; c5 += 1)
		                for (int c6 = c2; c6 <= c2 + 63; c6 += 1)
		                  Stmt_for_body8(c4, c6, c5);
		    }

		else
		    {  /* original code */ }
		[...]


	**Interchange + Tiling + Strip-mining to prepare vectorization**

	To later allow vectorization we create a so called trivially
        parallelizable loop. It is innermost, parallel and has only four
        iterations. It can be replaced by 4-element SIMD instructions.

	.. code-block:: console

		$ opt -basic-aa -polly-use-llvm-names matmul.preopt.ll -polly-import-jscop -polly-import-jscop-postfix=interchanged+tiled -polly-ast -analyze -polly-process-unprofitable

	.. code-block:: c

		[...]
		:: isl ast :: main :: %for.cond1.preheader---%for.end30

		if (1)

		    {
		      for (int c1 = 0; c1 <= 1535; c1 += 1)
		        for (int c2 = 0; c2 <= 1535; c2 += 1)
		          Stmt_for_body3(c1, c2);
		      for (int c1 = 0; c1 <= 1535; c1 += 64)
		        for (int c2 = 0; c2 <= 1535; c2 += 64)
		          for (int c3 = 0; c3 <= 1535; c3 += 64)
		            for (int c4 = c1; c4 <= c1 + 63; c4 += 1)
		              for (int c5 = c3; c5 <= c3 + 63; c5 += 1)
		                for (int c6 = c2; c6 <= c2 + 63; c6 += 4)
		                  for (int c7 = c6; c7 <= c6 + 3; c7 += 1)
		                    Stmt_for_body8(c4, c7, c5);
		    }

		else
		    {  /* original code */ }
		[...]

9. **Codegenerate the SCoPs**
-----------------------------

	This generates new code for the SCoPs detected by polly. If
        -polly-import-jscop is present, transformations specified in the
        imported jscop files will be applied.


	.. code-block:: console

		$ opt -S matmul.preopt.ll | opt -S -O3 -o matmul.normalopt.ll
		
	.. code-block:: console

		$ opt -S matmul.preopt.ll -basic-aa -polly-use-llvm-names -polly-import-jscop -polly-import-jscop-postfix=interchanged -polly-codegen -polly-process-unprofitable | opt -S -O3 -o matmul.polly.interchanged.ll

	.. code-block:: guess

		Reading JScop '%for.cond1.preheader---%for.end19' in function 'init_array' from './init_array___%for.cond1.preheader---%for.end19.jscop.interchanged'.
		File could not be read: No such file or directory
		Reading JScop '%for.cond1.preheader---%for.end30' in function 'main' from './main___%for.cond1.preheader---%for.end30.jscop.interchanged'.

	.. code-block:: console

		$ opt -S matmul.preopt.ll -basic-aa -polly-use-llvm-names -polly-import-jscop -polly-import-jscop-postfix=interchanged+tiled -polly-codegen -polly-process-unprofitable | opt -S -O3 -o matmul.polly.interchanged+tiled.ll
		
	.. code-block:: guess

		Reading JScop '%for.cond1.preheader---%for.end19' in function 'init_array' from './init_array___%for.cond1.preheader---%for.end19.jscop.interchanged+tiled'.
		File could not be read: No such file or directory
		Reading JScop '%for.cond1.preheader---%for.end30' in function 'main' from './main___%for.cond1.preheader---%for.end30.jscop.interchanged+tiled'.

	.. code-block:: console

		$ opt -S matmul.preopt.ll -basic-aa -polly-use-llvm-names -polly-import-jscop -polly-import-jscop-postfix=interchanged+tiled+vector -polly-codegen -polly-vectorizer=polly -polly-process-unprofitable | opt -S -O3 -o matmul.polly.interchanged+tiled+vector.ll

	.. code-block:: guess

		Reading JScop '%for.cond1.preheader---%for.end19' in function 'init_array' from './init_array___%for.cond1.preheader---%for.end19.jscop.interchanged+tiled+vector'.
		File could not be read: No such file or directory
		Reading JScop '%for.cond1.preheader---%for.end30' in function 'main' from './main___%for.cond1.preheader---%for.end30.jscop.interchanged+tiled+vector'.

	.. code-block:: console

		$ opt -S matmul.preopt.ll -basic-aa -polly-use-llvm-names -polly-import-jscop -polly-import-jscop-postfix=interchanged+tiled+vector -polly-codegen -polly-vectorizer=polly -polly-parallel -polly-process-unprofitable | opt -S -O3 -o matmul.polly.interchanged+tiled+openmp.ll

	.. code-block:: guess

		Reading JScop '%for.cond1.preheader---%for.end19' in function 'init_array' from './init_array___%for.cond1.preheader---%for.end19.jscop.interchanged+tiled+vector'.
		File could not be read: No such file or directory
		Reading JScop '%for.cond1.preheader---%for.end30' in function 'main' from './main___%for.cond1.preheader---%for.end30.jscop.interchanged+tiled+vector'.


10. **Create the executables**
------------------------------

        .. code-block:: console

	        $ llc matmul.normalopt.ll -o matmul.normalopt.s -relocation-model=pic
	        $ gcc matmul.normalopt.s -o matmul.normalopt.exe
	        $ llc matmul.polly.interchanged.ll -o matmul.polly.interchanged.s -relocation-model=pic
	        $ gcc matmul.polly.interchanged.s -o matmul.polly.interchanged.exe
	        $ llc matmul.polly.interchanged+tiled.ll -o matmul.polly.interchanged+tiled.s -relocation-model=pic
	        $ gcc matmul.polly.interchanged+tiled.s -o matmul.polly.interchanged+tiled.exe
	        $ llc matmul.polly.interchanged+tiled+vector.ll -o matmul.polly.interchanged+tiled+vector.s -relocation-model=pic
	        $ gcc matmul.polly.interchanged+tiled+vector.s -o matmul.polly.interchanged+tiled+vector.exe
        	$ llc matmul.polly.interchanged+tiled+vector+openmp.ll -o matmul.polly.interchanged+tiled+vector+openmp.s -relocation-model=pic
        	$ gcc matmul.polly.interchanged+tiled+vector+openmp.s -lgomp -o matmul.polly.interchanged+tiled+vector+openmp.exe

11. **Compare the runtime of the executables**
----------------------------------------------

	By comparing the runtimes of the different code snippets we see that a
        simple loop interchange gives here the largest performance boost.
        However in this case, adding vectorization and using OpenMP degrades
        the performance.

        .. code-block:: console

	        $ time ./matmul.normalopt.exe

	        real	0m11.295s
        	user	0m11.288s
        	sys	0m0.004s
        	$ time ./matmul.polly.interchanged.exe

        	real	0m0.988s
	        user	0m0.980s
	        sys	0m0.008s
	        $ time ./matmul.polly.interchanged+tiled.exe

	        real	0m0.830s
	        user	0m0.816s
	        sys	0m0.012s
	        $ time ./matmul.polly.interchanged+tiled+vector.exe

        	real	0m5.430s
        	user	0m5.424s
        	sys	0m0.004s
        	$ time ./matmul.polly.interchanged+tiled+vector+openmp.exe

        	real	0m3.184s
        	user	0m11.972s
        	sys	0m0.036s

