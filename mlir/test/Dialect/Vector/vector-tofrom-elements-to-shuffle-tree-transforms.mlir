// RUN: mlir-opt -lower-vector-to-from-elements-to-shuffle-tree -split-input-file %s | FileCheck %s

// Captured variable names for `vector.shuffle` operations follow the L#SH# convention,
// where L# refers to the level of the tree the shuffle belongs to, and SH# refers to
// the shuffle index within that level.

func.func @unsupported_trivial_forwarding(%a: vector<8xf32>) -> vector<8xf32> {
  %0:8 = vector.to_elements %a : vector<8xf32>
  %1 = vector.from_elements %0#0, %0#1, %0#2, %0#3, %0#4, %0#5, %0#6, %0#7 : vector<8xf32>
  return %1 : vector<8xf32>
}

// No shuffle tree needed for trivial forwarding case.

// CHECK-LABEL: func @unsupported_trivial_forwarding(
//  CHECK-SAME:     %[[A:.*]]: vector<8xf32>
//       CHECK:   return %[[A]] : vector<8xf32>

// -----

func.func @unsupported_multi_dim_vector_inputs(%a: vector<2x4xf32>, %b: vector<2x4xf32>) -> vector<4xf32> {
  %0:8 = vector.to_elements %a : vector<2x4xf32>
  %1:8 = vector.to_elements %b : vector<2x4xf32>
  %2 = vector.from_elements %0#0, %0#7,
                            %1#0, %1#7 : vector<4xf32>
  return %2 : vector<4xf32>
}

//   CHECK-LABEL: func @unsupported_multi_dim_vector_inputs(
// CHECK-COUNT-2:   vector.to_elements
//         CHECK:   vector.from_elements

// -----

func.func @unsupported_multi_dim_vector_output(%a: vector<8xf32>, %b: vector<8xf32>) -> vector<2x2xf32> {
  %0:8 = vector.to_elements %a : vector<8xf32>
  %1:8 = vector.to_elements %b : vector<8xf32>
  %2 = vector.from_elements %0#0, %0#7,
                            %1#0, %1#7 : vector<2x2xf32>
  return %2 : vector<2x2xf32>
}

// CHECK-LABEL: func @unsupported_multi_dim_vector_output(
// CHECK-COUNT-2:   vector.to_elements
//         CHECK:   vector.from_elements

// -----

func.func @shuffle_tree_single_input_shuffle(%a: vector<8xf32>) -> vector<8xf32> {
  %0:8 = vector.to_elements %a : vector<8xf32>
  %1 = vector.from_elements %0#7, %0#0, %0#6, %0#1, %0#5, %0#2, %0#4, %0#3 : vector<8xf32>
  return %1 : vector<8xf32>
}

// CHECK-LABEL: func @shuffle_tree_single_input_shuffle(
//  CHECK-SAME:     %[[A:.*]]: vector<8xf32>
      // CHECK:   %[[L0SH0:.*]] = vector.shuffle %[[A]], %[[A]] [7, 0, 6, 1, 5, 2, 4, 3] : vector<8xf32>, vector<8xf32>
      // CHECK:   return %[[L0SH0]]

// -----

func.func @shuffle_tree_single_shuffle(%a: vector<8xf32>,
                          %b: vector<8xf32>) -> vector<8xf32> {
  %0:8 = vector.to_elements %a : vector<8xf32>
  %1:8 = vector.to_elements %b : vector<8xf32>
  %2 = vector.from_elements %0#7, %1#0, %0#6, %1#1, %0#5, %1#2, %0#4, %1#3 : vector<8xf32>
  return %2 : vector<8xf32>
}

// CHECK-LABEL: func @shuffle_tree_single_shuffle(
//  CHECK-SAME:     %[[A:.*]]: vector<8xf32>, %[[B:.*]]: vector<8xf32>
//       CHECK:   %[[L0SH0:.*]] = vector.shuffle %[[A]], %[[B]] [7, 8, 6, 9, 5, 10, 4, 11] : vector<8xf32>
//       CHECK:   return %[[L0SH0]]

// -----

func.func @shuffle_tree_concat_4x8_to_32(%a: vector<8xf32>,
                                         %b: vector<8xf32>,
                                         %c: vector<8xf32>,
                                         %d: vector<8xf32>) -> vector<32xf32> {
  %0:8 = vector.to_elements %a : vector<8xf32>
  %1:8 = vector.to_elements %b : vector<8xf32>
  %2:8 = vector.to_elements %c : vector<8xf32>
  %3:8 = vector.to_elements %d : vector<8xf32>
  %4 = vector.from_elements %0#0, %0#1, %0#2, %0#3, %0#4, %0#5, %0#6, %0#7,
                            %1#0, %1#1, %1#2, %1#3, %1#4, %1#5, %1#6, %1#7,
                            %2#0, %2#1, %2#2, %2#3, %2#4, %2#5, %2#6, %2#7,
                            %3#0, %3#1, %3#2, %3#3, %3#4, %3#5, %3#6, %3#7 : vector<32xf32>
  return %4 : vector<32xf32>
}

// CHECK-LABEL: func @shuffle_tree_concat_4x8_to_32(
//  CHECK-SAME:     %[[A:.*]]: vector<8xf32>, %[[B:.*]]: vector<8xf32>, %[[C:.*]]: vector<8xf32>, %[[D:.*]]: vector<8xf32>
//       CHECK:   %[[L0SH0:.*]] = vector.shuffle %[[A]], %[[B]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] : vector<8xf32>, vector<8xf32>
//       CHECK:   %[[L0SH1:.*]] = vector.shuffle %[[C]], %[[D]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] : vector<8xf32>, vector<8xf32>
//       CHECK:   %[[L1SH0:.*]] = vector.shuffle %[[L0SH0]], %[[L0SH1]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] : vector<16xf32>, vector<16xf32>
//       CHECK:   return %[[L1SH0]] : vector<32xf32>

// -----

func.func @shuffle_tree_concat_3x4_to_12(%a: vector<4xf32>,
                                                          %b: vector<4xf32>,
                                                          %c: vector<4xf32>) -> vector<12xf32> {
  %0:4 = vector.to_elements %a : vector<4xf32>
  %1:4 = vector.to_elements %b : vector<4xf32>
  %2:4 = vector.to_elements %c : vector<4xf32>
  %3 = vector.from_elements %0#0, %0#1, %0#2, %0#3, %1#0, %1#1, %1#2, %1#3, %2#0, %2#1, %2#2, %2#3 : vector<12xf32>
  return %3 : vector<12xf32>
}

// CHECK-LABEL: func @shuffle_tree_concat_3x4_to_12(
//  CHECK-SAME:     %[[A:.*]]: vector<4xf32>, %[[B:.*]]: vector<4xf32>, %[[C:.*]]: vector<4xf32>
//       CHECK:   %[[L0SH0:.*]] = vector.shuffle %[[A]], %[[B]] [0, 1, 2, 3, 4, 5, 6, 7] : vector<4xf32>, vector<4xf32>
//       CHECK:   %[[L0SH1:.*]] = vector.shuffle %[[C]], %[[C]] [0, 1, 2, 3, -1, -1, -1, -1] : vector<4xf32>, vector<4xf32>
//       CHECK:   %[[L1SH0:.*]] = vector.shuffle %[[L0SH0]], %[[L0SH1]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11] : vector<8xf32>, vector<8xf32>
//       CHECK:   return %[[L1SH0]] : vector<12xf32>

// -----

func.func @shuffle_tree_concat_64x4_256(%a: vector<4xf32>, %b: vector<4xf32>, %c: vector<4xf32>, %d: vector<4xf32>,
                                        %e: vector<4xf32>, %f: vector<4xf32>, %g: vector<4xf32>, %h: vector<4xf32>,
                                        %i: vector<4xf32>, %j: vector<4xf32>, %k: vector<4xf32>, %l: vector<4xf32>,
                                        %m: vector<4xf32>, %n: vector<4xf32>, %o: vector<4xf32>, %p: vector<4xf32>,
                                        %q: vector<4xf32>, %r: vector<4xf32>, %s: vector<4xf32>, %t: vector<4xf32>,
                                        %u: vector<4xf32>, %v: vector<4xf32>, %w: vector<4xf32>, %x: vector<4xf32>,
                                        %y: vector<4xf32>, %z: vector<4xf32>, %aa: vector<4xf32>, %ab: vector<4xf32>,
                                        %ac: vector<4xf32>, %ad: vector<4xf32>, %ae: vector<4xf32>, %af: vector<4xf32>,
                                        %ag: vector<4xf32>, %ah: vector<4xf32>, %ai: vector<4xf32>, %aj: vector<4xf32>,
                                        %ak: vector<4xf32>, %al: vector<4xf32>, %am: vector<4xf32>, %an: vector<4xf32>,
                                        %ao: vector<4xf32>, %ap: vector<4xf32>, %aq: vector<4xf32>, %ar: vector<4xf32>,
                                        %as: vector<4xf32>, %at: vector<4xf32>, %au: vector<4xf32>, %av: vector<4xf32>,
                                        %aw: vector<4xf32>, %ax: vector<4xf32>, %ay: vector<4xf32>, %az: vector<4xf32>,
                                        %ba: vector<4xf32>, %bb: vector<4xf32>, %bc: vector<4xf32>, %bd: vector<4xf32>,
                                        %be: vector<4xf32>, %bf: vector<4xf32>, %bg: vector<4xf32>, %bh: vector<4xf32>,
                                        %bi: vector<4xf32>, %bj: vector<4xf32>, %bk: vector<4xf32>, %bl: vector<4xf32>) -> vector<256xf32> {
  %0:4 = vector.to_elements %a : vector<4xf32>
  %1:4 = vector.to_elements %b : vector<4xf32>
  %2:4 = vector.to_elements %c : vector<4xf32>
  %3:4 = vector.to_elements %d : vector<4xf32>
  %4:4 = vector.to_elements %e : vector<4xf32>
  %5:4 = vector.to_elements %f : vector<4xf32>
  %6:4 = vector.to_elements %g : vector<4xf32>
  %7:4 = vector.to_elements %h : vector<4xf32>
  %8:4 = vector.to_elements %i : vector<4xf32>
  %9:4 = vector.to_elements %j : vector<4xf32>
  %10:4 = vector.to_elements %k : vector<4xf32>
  %11:4 = vector.to_elements %l : vector<4xf32>
  %12:4 = vector.to_elements %m : vector<4xf32>
  %13:4 = vector.to_elements %n : vector<4xf32>
  %14:4 = vector.to_elements %o : vector<4xf32>
  %15:4 = vector.to_elements %p : vector<4xf32>
  %16:4 = vector.to_elements %q : vector<4xf32>
  %17:4 = vector.to_elements %r : vector<4xf32>
  %18:4 = vector.to_elements %s : vector<4xf32>
  %19:4 = vector.to_elements %t : vector<4xf32>
  %20:4 = vector.to_elements %u : vector<4xf32>
  %21:4 = vector.to_elements %v : vector<4xf32>
  %22:4 = vector.to_elements %w : vector<4xf32>
  %23:4 = vector.to_elements %x : vector<4xf32>
  %24:4 = vector.to_elements %y : vector<4xf32>
  %25:4 = vector.to_elements %z : vector<4xf32>
  %26:4 = vector.to_elements %aa : vector<4xf32>
  %27:4 = vector.to_elements %ab : vector<4xf32>
  %28:4 = vector.to_elements %ac : vector<4xf32>
  %29:4 = vector.to_elements %ad : vector<4xf32>
  %30:4 = vector.to_elements %ae : vector<4xf32>
  %31:4 = vector.to_elements %af : vector<4xf32>
  %32:4 = vector.to_elements %ag : vector<4xf32>
  %33:4 = vector.to_elements %ah : vector<4xf32>
  %34:4 = vector.to_elements %ai : vector<4xf32>
  %35:4 = vector.to_elements %aj : vector<4xf32>
  %36:4 = vector.to_elements %ak : vector<4xf32>
  %37:4 = vector.to_elements %al : vector<4xf32>
  %38:4 = vector.to_elements %am : vector<4xf32>
  %39:4 = vector.to_elements %an : vector<4xf32>
  %40:4 = vector.to_elements %ao : vector<4xf32>
  %41:4 = vector.to_elements %ap : vector<4xf32>
  %42:4 = vector.to_elements %aq : vector<4xf32>
  %43:4 = vector.to_elements %ar : vector<4xf32>
  %44:4 = vector.to_elements %as : vector<4xf32>
  %45:4 = vector.to_elements %at : vector<4xf32>
  %46:4 = vector.to_elements %au : vector<4xf32>
  %47:4 = vector.to_elements %av : vector<4xf32>
  %48:4 = vector.to_elements %aw : vector<4xf32>
  %49:4 = vector.to_elements %ax : vector<4xf32>
  %50:4 = vector.to_elements %ay : vector<4xf32>
  %51:4 = vector.to_elements %az : vector<4xf32>
  %52:4 = vector.to_elements %ba : vector<4xf32>
  %53:4 = vector.to_elements %bb : vector<4xf32>
  %54:4 = vector.to_elements %bc : vector<4xf32>
  %55:4 = vector.to_elements %bd : vector<4xf32>
  %56:4 = vector.to_elements %be : vector<4xf32>
  %57:4 = vector.to_elements %bf : vector<4xf32>
  %58:4 = vector.to_elements %bg : vector<4xf32>
  %59:4 = vector.to_elements %bh : vector<4xf32>
  %60:4 = vector.to_elements %bi : vector<4xf32>
  %61:4 = vector.to_elements %bj : vector<4xf32>
  %62:4 = vector.to_elements %bk : vector<4xf32>
  %63:4 = vector.to_elements %bl : vector<4xf32>
  %64 = vector.from_elements %0#0, %0#1, %0#2, %0#3, %1#0, %1#1, %1#2, %1#3, %2#0, %2#1, %2#2, %2#3, %3#0, %3#1, %3#2, %3#3, %4#0, %4#1, %4#2, %4#3,
                             %5#0, %5#1, %5#2, %5#3, %6#0, %6#1, %6#2, %6#3, %7#0, %7#1, %7#2, %7#3, %8#0, %8#1, %8#2, %8#3, %9#0, %9#1, %9#2, %9#3,
                             %10#0, %10#1, %10#2, %10#3, %11#0, %11#1, %11#2, %11#3, %12#0, %12#1, %12#2, %12#3, %13#0, %13#1, %13#2, %13#3, %14#0, %14#1, %14#2, %14#3,
                             %15#0, %15#1, %15#2, %15#3, %16#0, %16#1, %16#2, %16#3, %17#0, %17#1, %17#2, %17#3, %18#0, %18#1, %18#2, %18#3, %19#0, %19#1, %19#2, %19#3,
                             %20#0, %20#1, %20#2, %20#3, %21#0, %21#1, %21#2, %21#3, %22#0, %22#1, %22#2, %22#3, %23#0, %23#1, %23#2, %23#3, %24#0, %24#1, %24#2, %24#3,
                             %25#0, %25#1, %25#2, %25#3, %26#0, %26#1, %26#2, %26#3, %27#0, %27#1, %27#2, %27#3, %28#0, %28#1, %28#2, %28#3, %29#0, %29#1, %29#2, %29#3,
                             %30#0, %30#1, %30#2, %30#3, %31#0, %31#1, %31#2, %31#3, %32#0, %32#1, %32#2, %32#3, %33#0, %33#1, %33#2, %33#3, %34#0, %34#1, %34#2, %34#3,
                             %35#0, %35#1, %35#2, %35#3, %36#0, %36#1, %36#2, %36#3, %37#0, %37#1, %37#2, %37#3, %38#0, %38#1, %38#2, %38#3, %39#0, %39#1, %39#2, %39#3,
                             %40#0, %40#1, %40#2, %40#3, %41#0, %41#1, %41#2, %41#3, %42#0, %42#1, %42#2, %42#3, %43#0, %43#1, %43#2, %43#3, %44#0, %44#1, %44#2, %44#3,
                             %45#0, %45#1, %45#2, %45#3, %46#0, %46#1, %46#2, %46#3, %47#0, %47#1, %47#2, %47#3, %48#0, %48#1, %48#2, %48#3, %49#0, %49#1, %49#2, %49#3,
                             %50#0, %50#1, %50#2, %50#3, %51#0, %51#1, %51#2, %51#3, %52#0, %52#1, %52#2, %52#3, %53#0, %53#1, %53#2, %53#3, %54#0, %54#1, %54#2, %54#3,
                             %55#0, %55#1, %55#2, %55#3, %56#0, %56#1, %56#2, %56#3, %57#0, %57#1, %57#2, %57#3, %58#0, %58#1, %58#2, %58#3, %59#0, %59#1, %59#2, %59#3,
                             %60#0, %60#1, %60#2, %60#3, %61#0, %61#1, %61#2, %61#3, %62#0, %62#1, %62#2, %62#3, %63#0, %63#1, %63#2, %63#3 : vector<256xf32>
  return %64 : vector<256xf32>
}

// CHECK-LABEL: func.func @shuffle_tree_concat_64x4_256(
//  CHECK-SAME:     %[[A:.+]]: vector<4xf32>, %[[B:.+]]: vector<4xf32>, %[[C:.+]]: vector<4xf32>, %[[D:.+]]: vector<4xf32>, %[[E:.+]]: vector<4xf32>, %[[F:.+]]: vector<4xf32>, %[[G:.+]]: vector<4xf32>, %[[H:.+]]: vector<4xf32>, %[[I:.+]]: vector<4xf32>, %[[J:.+]]: vector<4xf32>, %[[K:.+]]: vector<4xf32>, %[[L:.+]]: vector<4xf32>, %[[M:.+]]: vector<4xf32>, %[[N:.+]]: vector<4xf32>, %[[O:.+]]: vector<4xf32>, %[[P:.+]]: vector<4xf32>, %[[Q:.+]]: vector<4xf32>, %[[R:.+]]: vector<4xf32>, %[[S:.+]]: vector<4xf32>, %[[T:.+]]: vector<4xf32>, %[[U:.+]]: vector<4xf32>, %[[V:.+]]: vector<4xf32>, %[[W:.+]]: vector<4xf32>, %[[X:.+]]: vector<4xf32>, %[[Y:.+]]: vector<4xf32>, %[[Z:.+]]: vector<4xf32>, %[[AA:.+]]: vector<4xf32>, %[[AB:.+]]: vector<4xf32>, %[[AC:.+]]: vector<4xf32>, %[[AD:.+]]: vector<4xf32>, %[[AE:.+]]: vector<4xf32>, %[[AF:.+]]: vector<4xf32>, %[[AG:.+]]: vector<4xf32>, %[[AH:.+]]: vector<4xf32>, %[[AI:.+]]: vector<4xf32>, %[[AJ:.+]]: vector<4xf32>, %[[AK:.+]]: vector<4xf32>, %[[AL:.+]]: vector<4xf32>, %[[AM:.+]]: vector<4xf32>, %[[AN:.+]]: vector<4xf32>, %[[AO:.+]]: vector<4xf32>, %[[AP:.+]]: vector<4xf32>, %[[AQ:.+]]: vector<4xf32>, %[[AR:.+]]: vector<4xf32>, %[[AS:.+]]: vector<4xf32>, %[[AT:.+]]: vector<4xf32>, %[[AU:.+]]: vector<4xf32>, %[[AV:.+]]: vector<4xf32>, %[[AW:.+]]: vector<4xf32>, %[[AX:.+]]: vector<4xf32>, %[[AY:.+]]: vector<4xf32>, %[[AZ:.+]]: vector<4xf32>, %[[BA:.+]]: vector<4xf32>, %[[BB:.+]]: vector<4xf32>, %[[BC:.+]]: vector<4xf32>, %[[BD:.+]]: vector<4xf32>, %[[BE:.+]]: vector<4xf32>, %[[BF:.+]]: vector<4xf32>, %[[BG:.+]]: vector<4xf32>, %[[BH:.+]]: vector<4xf32>, %[[BI:.+]]: vector<4xf32>, %[[BJ:.+]]: vector<4xf32>, %[[BK:.+]]: vector<4xf32>, %[[BL:.+]]: vector<4xf32>)
//       CHECK:   %[[L0SH0:.+]] = vector.shuffle %[[A]], %[[B]] [0, 1, 2, 3, 4, 5, 6, 7] : vector<4xf32>, vector<4xf32>
//       CHECK:   %[[L0SH1:.+]] = vector.shuffle %[[C]], %[[D]] [0, 1, 2, 3, 4, 5, 6, 7] : vector<4xf32>, vector<4xf32>
//       CHECK:   %[[L0SH2:.+]] = vector.shuffle %[[E]], %[[F]] [0, 1, 2, 3, 4, 5, 6, 7] : vector<4xf32>, vector<4xf32>
//       CHECK:   %[[L0SH3:.+]] = vector.shuffle %[[G]], %[[H]] [0, 1, 2, 3, 4, 5, 6, 7] : vector<4xf32>, vector<4xf32>
//       CHECK:   %[[L0SH4:.+]] = vector.shuffle %[[I]], %[[J]] [0, 1, 2, 3, 4, 5, 6, 7] : vector<4xf32>, vector<4xf32>
//       CHECK:   %[[L0SH5:.+]] = vector.shuffle %[[K]], %[[L]] [0, 1, 2, 3, 4, 5, 6, 7] : vector<4xf32>, vector<4xf32>
//       CHECK:   %[[L0SH6:.+]] = vector.shuffle %[[M]], %[[N]] [0, 1, 2, 3, 4, 5, 6, 7] : vector<4xf32>, vector<4xf32>
//       CHECK:   %[[L0SH7:.+]] = vector.shuffle %[[O]], %[[P]] [0, 1, 2, 3, 4, 5, 6, 7] : vector<4xf32>, vector<4xf32>
//       CHECK:   %[[L0SH8:.+]] = vector.shuffle %[[Q]], %[[R]] [0, 1, 2, 3, 4, 5, 6, 7] : vector<4xf32>, vector<4xf32>
//       CHECK:   %[[L0SH9:.+]] = vector.shuffle %[[S]], %[[T]] [0, 1, 2, 3, 4, 5, 6, 7] : vector<4xf32>, vector<4xf32>
//       CHECK:   %[[L0SH10:.+]] = vector.shuffle %[[U]], %[[V]] [0, 1, 2, 3, 4, 5, 6, 7] : vector<4xf32>, vector<4xf32>
//       CHECK:   %[[L0SH11:.+]] = vector.shuffle %[[W]], %[[X]] [0, 1, 2, 3, 4, 5, 6, 7] : vector<4xf32>, vector<4xf32>
//       CHECK:   %[[L0SH12:.+]] = vector.shuffle %[[Y]], %[[Z]] [0, 1, 2, 3, 4, 5, 6, 7] : vector<4xf32>, vector<4xf32>
//       CHECK:   %[[L0SH13:.+]] = vector.shuffle %[[AA]], %[[AB]] [0, 1, 2, 3, 4, 5, 6, 7] : vector<4xf32>, vector<4xf32>
//       CHECK:   %[[L0SH14:.+]] = vector.shuffle %[[AC]], %[[AD]] [0, 1, 2, 3, 4, 5, 6, 7] : vector<4xf32>, vector<4xf32>
//       CHECK:   %[[L0SH15:.+]] = vector.shuffle %[[AE]], %[[AF]] [0, 1, 2, 3, 4, 5, 6, 7] : vector<4xf32>, vector<4xf32>
//       CHECK:   %[[L0SH16:.+]] = vector.shuffle %[[AG]], %[[AH]] [0, 1, 2, 3, 4, 5, 6, 7] : vector<4xf32>, vector<4xf32>
//       CHECK:   %[[L0SH17:.+]] = vector.shuffle %[[AI]], %[[AJ]] [0, 1, 2, 3, 4, 5, 6, 7] : vector<4xf32>, vector<4xf32>
//       CHECK:   %[[L0SH18:.+]] = vector.shuffle %[[AK]], %[[AL]] [0, 1, 2, 3, 4, 5, 6, 7] : vector<4xf32>, vector<4xf32>
//       CHECK:   %[[L0SH19:.+]] = vector.shuffle %[[AM]], %[[AN]] [0, 1, 2, 3, 4, 5, 6, 7] : vector<4xf32>, vector<4xf32>
//       CHECK:   %[[L0SH20:.+]] = vector.shuffle %[[AO]], %[[AP]] [0, 1, 2, 3, 4, 5, 6, 7] : vector<4xf32>, vector<4xf32>
//       CHECK:   %[[L0SH21:.+]] = vector.shuffle %[[AQ]], %[[AR]] [0, 1, 2, 3, 4, 5, 6, 7] : vector<4xf32>, vector<4xf32>
//       CHECK:   %[[L0SH22:.+]] = vector.shuffle %[[AS]], %[[AT]] [0, 1, 2, 3, 4, 5, 6, 7] : vector<4xf32>, vector<4xf32>
//       CHECK:   %[[L0SH23:.+]] = vector.shuffle %[[AU]], %[[AV]] [0, 1, 2, 3, 4, 5, 6, 7] : vector<4xf32>, vector<4xf32>
//       CHECK:   %[[L0SH24:.+]] = vector.shuffle %[[AW]], %[[AX]] [0, 1, 2, 3, 4, 5, 6, 7] : vector<4xf32>, vector<4xf32>
//       CHECK:   %[[L0SH25:.+]] = vector.shuffle %[[AY]], %[[AZ]] [0, 1, 2, 3, 4, 5, 6, 7] : vector<4xf32>, vector<4xf32>
//       CHECK:   %[[L0SH26:.+]] = vector.shuffle %[[BA]], %[[BB]] [0, 1, 2, 3, 4, 5, 6, 7] : vector<4xf32>, vector<4xf32>
//       CHECK:   %[[L0SH27:.+]] = vector.shuffle %[[BC]], %[[BD]] [0, 1, 2, 3, 4, 5, 6, 7] : vector<4xf32>, vector<4xf32>
//       CHECK:   %[[L0SH28:.+]] = vector.shuffle %[[BE]], %[[BF]] [0, 1, 2, 3, 4, 5, 6, 7] : vector<4xf32>, vector<4xf32>
//       CHECK:   %[[L0SH29:.+]] = vector.shuffle %[[BG]], %[[BH]] [0, 1, 2, 3, 4, 5, 6, 7] : vector<4xf32>, vector<4xf32>
//       CHECK:   %[[L0SH30:.+]] = vector.shuffle %[[BI]], %[[BJ]] [0, 1, 2, 3, 4, 5, 6, 7] : vector<4xf32>, vector<4xf32>
//       CHECK:   %[[L0SH31:.+]] = vector.shuffle %[[BK]], %[[BL]] [0, 1, 2, 3, 4, 5, 6, 7] : vector<4xf32>, vector<4xf32>
//       CHECK:   %[[L1SH0:.+]] = vector.shuffle %[[L0SH0]], %[[L0SH1]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] : vector<8xf32>, vector<8xf32>
//       CHECK:   %[[L1SH1:.+]] = vector.shuffle %[[L0SH2]], %[[L0SH3]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] : vector<8xf32>, vector<8xf32>
//       CHECK:   %[[L1SH2:.+]] = vector.shuffle %[[L0SH4]], %[[L0SH5]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] : vector<8xf32>, vector<8xf32>
//       CHECK:   %[[L1SH3:.+]] = vector.shuffle %[[L0SH6]], %[[L0SH7]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] : vector<8xf32>, vector<8xf32>
//       CHECK:   %[[L1SH4:.+]] = vector.shuffle %[[L0SH8]], %[[L0SH9]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] : vector<8xf32>, vector<8xf32>
//       CHECK:   %[[L1SH5:.+]] = vector.shuffle %[[L0SH10]], %[[L0SH11]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] : vector<8xf32>, vector<8xf32>
//       CHECK:   %[[L1SH6:.+]] = vector.shuffle %[[L0SH12]], %[[L0SH13]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] : vector<8xf32>, vector<8xf32>
//       CHECK:   %[[L1SH7:.+]] = vector.shuffle %[[L0SH14]], %[[L0SH15]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] : vector<8xf32>, vector<8xf32>
//       CHECK:   %[[L1SH8:.+]] = vector.shuffle %[[L0SH16]], %[[L0SH17]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] : vector<8xf32>, vector<8xf32>
//       CHECK:   %[[L1SH9:.+]] = vector.shuffle %[[L0SH18]], %[[L0SH19]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] : vector<8xf32>, vector<8xf32>
//       CHECK:   %[[L1SH10:.+]] = vector.shuffle %[[L0SH20]], %[[L0SH21]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] : vector<8xf32>, vector<8xf32>
//       CHECK:   %[[L1SH11:.+]] = vector.shuffle %[[L0SH22]], %[[L0SH23]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] : vector<8xf32>, vector<8xf32>
//       CHECK:   %[[L1SH12:.+]] = vector.shuffle %[[L0SH24]], %[[L0SH25]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] : vector<8xf32>, vector<8xf32>
//       CHECK:   %[[L1SH13:.+]] = vector.shuffle %[[L0SH26]], %[[L0SH27]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] : vector<8xf32>, vector<8xf32>
//       CHECK:   %[[L1SH14:.+]] = vector.shuffle %[[L0SH28]], %[[L0SH29]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] : vector<8xf32>, vector<8xf32>
//       CHECK:   %[[L1SH15:.+]] = vector.shuffle %[[L0SH30]], %[[L0SH31]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] : vector<8xf32>, vector<8xf32>
//       CHECK:   %[[L2SH0:.+]] = vector.shuffle %[[L1SH0]], %[[L1SH1]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] : vector<16xf32>, vector<16xf32>
//       CHECK:   %[[L2SH1:.+]] = vector.shuffle %[[L1SH2]], %[[L1SH3]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] : vector<16xf32>, vector<16xf32>
//       CHECK:   %[[L2SH2:.+]] = vector.shuffle %[[L1SH4]], %[[L1SH5]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] : vector<16xf32>, vector<16xf32>
//       CHECK:   %[[L2SH3:.+]] = vector.shuffle %[[L1SH6]], %[[L1SH7]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] : vector<16xf32>, vector<16xf32>
//       CHECK:   %[[L2SH4:.+]] = vector.shuffle %[[L1SH8]], %[[L1SH9]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] : vector<16xf32>, vector<16xf32>
//       CHECK:   %[[L2SH5:.+]] = vector.shuffle %[[L1SH10]], %[[L1SH11]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] : vector<16xf32>, vector<16xf32>
//       CHECK:   %[[L2SH6:.+]] = vector.shuffle %[[L1SH12]], %[[L1SH13]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] : vector<16xf32>, vector<16xf32>
//       CHECK:   %[[L2SH7:.+]] = vector.shuffle %[[L1SH14]], %[[L1SH15]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] : vector<16xf32>, vector<16xf32>
//       CHECK:   %[[L3SH0:.+]] = vector.shuffle %[[L2SH0]], %[[L2SH1]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63] : vector<32xf32>, vector<32xf32>
//       CHECK:   %[[L3SH1:.+]] = vector.shuffle %[[L2SH2]], %[[L2SH3]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63] : vector<32xf32>, vector<32xf32>
//       CHECK:   %[[L3SH2:.+]] = vector.shuffle %[[L2SH4]], %[[L2SH5]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63] : vector<32xf32>, vector<32xf32>
//       CHECK:   %[[L3SH3:.+]] = vector.shuffle %[[L2SH6]], %[[L2SH7]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63] : vector<32xf32>, vector<32xf32>
//       CHECK:   %[[L4SH0:.+]] = vector.shuffle %[[L3SH0]], %[[L3SH1]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127] : vector<64xf32>, vector<64xf32>
//       CHECK:   %[[L4SH1:.+]] = vector.shuffle %[[L3SH2]], %[[L3SH3]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127] : vector<64xf32>, vector<64xf32>
//       CHECK:   %[[L5SH0:.+]] = vector.shuffle %[[L4SH0]], %[[L4SH1]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255] : vector<128xf32>, vector<128xf32>
//       CHECK:   return %[[L5SH0]] : vector<256xf32>

// -----

func.func @shuffle_tree_arbitrary_4x4_to_16(%a: vector<4xf32>,
                                            %b: vector<4xf32>,
                                            %c: vector<4xf32>,
                                            %d: vector<4xf32>) -> vector<16xf32> {
  %0:4 = vector.to_elements %a : vector<4xf32>
  %1:4 = vector.to_elements %b : vector<4xf32>
  %2:4 = vector.to_elements %c : vector<4xf32>
  %3:4 = vector.to_elements %d : vector<4xf32>
  %4 = vector.from_elements %3#3, %0#0, %2#2, %1#1, %3#0, %2#1, %0#3, %1#2, %0#1, %3#2, %1#0, %2#3, %1#3, %0#2, %3#1, %2#0 : vector<16xf32>
  return %4 : vector<16xf32>
}

// TODO: Implement mask compression to reduce the number of intermediate poison values.

// CHECK-LABEL: func @shuffle_tree_arbitrary_4x4_to_16(
//  CHECK-SAME:     %[[A:.*]]: vector<4xf32>, %[[B:.*]]: vector<4xf32>, %[[C:.*]]: vector<4xf32>, %[[D:.*]]: vector<4xf32>
//       CHECK:   %[[L0SH0:.*]] = vector.shuffle %[[D]], %[[A]] [3, 4, -1, -1, 0, -1, 7, -1, 5, 2, -1, -1, -1, 6, 1] : vector<4xf32>, vector<4xf32>
//       CHECK:   %[[L0SH1:.*]] = vector.shuffle %[[C]], %[[B]] [2, 5, -1, 1, -1, 6, -1, -1, 4, 3, 7, -1, -1, 0, -1] : vector<4xf32>, vector<4xf32>
//       CHECK:   %[[L1SH0:.*]] = vector.shuffle %[[L0SH0]], %[[L0SH1]] [0, 1, 15, 16, 4, 18, 6, 20, 8, 9, 23, 24, 25, 13, 14, 28] : vector<15xf32>, vector<15xf32>
//       CHECK:   return %[[L1SH0]] : vector<16xf32>

// -----

func.func @shuffle_tree_arbitrary_3x4_to_12(%a: vector<4xf32>,
                                            %b: vector<4xf32>,
                                            %c: vector<4xf32>) -> vector<12xf32> {
  %0:4 = vector.to_elements %a : vector<4xf32>
  %1:4 = vector.to_elements %b : vector<4xf32>
  %2:4 = vector.to_elements %c : vector<4xf32>
  %3 = vector.from_elements %0#2, %1#1, %2#0, %0#1, %1#0, %2#2, %0#0, %1#3, %2#3, %0#3, %1#2, %2#1 : vector<12xf32>
  return %3 : vector<12xf32>
}

// TODO: Implement mask compression to reduce the number of intermediate poison values.

// CHECK-LABEL: func @shuffle_tree_arbitrary_3x4_to_12(
//  CHECK-SAME:     %[[A:.*]]: vector<4xf32>, %[[B:.*]]: vector<4xf32>, %[[C:.*]]: vector<4xf32>
//       CHECK:   %[[L0SH0:.*]] = vector.shuffle %[[A]], %[[B]] [2, 5, -1, 1, 4, -1, 0, 7, -1, 3, 6] : vector<4xf32>, vector<4xf32>
//       CHECK:   %[[L0SH1:.*]] = vector.shuffle %[[C]], %[[C]] [0, -1, -1, 2, -1, -1, 3, -1, -1, 1, -1] : vector<4xf32>, vector<4xf32>
//       CHECK:   %[[L1SH0:.*]] = vector.shuffle %[[L0SH0]], %[[L0SH1]] [0, 1, 11, 3, 4, 14, 6, 7, 17, 9, 10, 20] : vector<11xf32>, vector<11xf32>
//       CHECK:   return %[[L1SH0]] : vector<12xf32>

// -----

func.func @shuffle_tree_arbitrary_3x5_to_9(%a: vector<5xf32>,
                                           %b: vector<5xf32>,
                                           %c: vector<5xf32>) -> vector<9xf32> {
  %0:5 = vector.to_elements %a : vector<5xf32>
  %1:5 = vector.to_elements %b : vector<5xf32>
  %2:5 = vector.to_elements %c : vector<5xf32>
  %3 = vector.from_elements %2#2, %1#1, %0#1, %0#1, %1#2, %2#2, %2#0, %1#1, %0#4 : vector<9xf32>
  return %3 : vector<9xf32>
}

// TODO: Implement mask compression to reduce the number of intermediate poison values.

// CHECK-LABEL: func @shuffle_tree_arbitrary_3x5_to_9(
//  CHECK-SAME:     %[[A:.*]]: vector<5xf32>, %[[B:.*]]: vector<5xf32>, %[[C:.*]]: vector<5xf32>
//       CHECK:   %[[L0SH0:.*]] = vector.shuffle %[[C]], %[[B]] [2, 6, -1, -1, 7, 2, 0, 6] : vector<5xf32>, vector<5xf32>
//       CHECK:   %[[L0SH1:.*]] = vector.shuffle %[[A]], %[[A]] [1, 1, -1, -1, -1, -1, 4, -1] : vector<5xf32>, vector<5xf32>
//       CHECK:   %[[L1SH0:.*]] = vector.shuffle %[[L0SH0]], %[[L0SH1]] [0, 1, 8, 9, 4, 5, 6, 7, 14] : vector<8xf32>, vector<8xf32>
//       CHECK:   return %[[L1SH0]] : vector<9xf32>

// -----

func.func @shuffle_tree_broadcast_4x2_to_32(%a: vector<2xf32>,
                                            %b: vector<2xf32>,
                                            %c: vector<2xf32>,
                                            %d: vector<2xf32>) -> vector<32xf32> {
  %0:2 = vector.to_elements %a : vector<2xf32>
  %1:2 = vector.to_elements %b : vector<2xf32>
  %2:2 = vector.to_elements %c : vector<2xf32>
  %3:2 = vector.to_elements %d : vector<2xf32>
  %4 = vector.from_elements %0#0, %0#0, %0#0, %0#0, %0#1, %0#1, %0#1, %0#1,
                            %1#0, %1#0, %1#0, %1#0, %1#1, %1#1, %1#1, %1#1,
                            %2#0, %2#0, %2#0, %2#0, %2#1, %2#1, %2#1, %2#1,
                            %3#0, %3#0, %3#0, %3#0, %3#1, %3#1, %3#1, %3#1 : vector<32xf32>
  return %4 : vector<32xf32>
}

// CHECK-LABEL: func @shuffle_tree_broadcast_4x2_to_32(
//  CHECK-SAME:     %[[A:.*]]: vector<2xf32>, %[[B:.*]]: vector<2xf32>, %[[C:.*]]: vector<2xf32>, %[[D:.*]]: vector<2xf32>
      // CHECK:   %[[L0SH0:.*]] = vector.shuffle %[[A]], %[[B]] [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3] : vector<2xf32>, vector<2xf32>
      // CHECK:   %[[L0SH1:.*]] = vector.shuffle %[[C]], %[[D]] [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3] : vector<2xf32>, vector<2xf32>
      // CHECK:   %[[L1SH0:.*]] = vector.shuffle %[[L0SH0]], %[[L0SH1]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] : vector<16xf32>, vector<16xf32>
      // CHECK:   return %[[L1SH0]] : vector<32xf32>

// -----

func.func @shuffle_tree_arbitrary_mixed_sizes(%a : vector<2xf32>,
                                              %b : vector<1xf32>,
                                              %c : vector<3xf32>,
                                              %d : vector<1xf32>,
                                              %e : vector<5xf32>) -> vector<6xf32> {
  %0:2 = vector.to_elements %a : vector<2xf32>
  %1 = vector.to_elements %b : vector<1xf32>
  %2:3 = vector.to_elements %c : vector<3xf32>
  %3 = vector.to_elements %d : vector<1xf32>
  %4:5 = vector.to_elements %e : vector<5xf32>
  %5 = vector.from_elements %0#0, %2#0, %3, %4#0, %1, %4#3 : vector<6xf32>
  return %5 : vector<6xf32>
}

//   CHECK-LABEL: func @shuffle_tree_arbitrary_mixed_sizes(
//  CHECK-SAME:     %[[A:.*]]: vector<2xf32>, %[[B:.*]]: vector<1xf32>, %[[C:.*]]: vector<3xf32>, %[[D:.*]]: vector<1xf32>, %[[E:.*]]: vector<5xf32>
//       CHECK:   %[[L0SH0:.*]] = vector.shuffle %[[A]], %[[C]] [0, 2, -1, -1] : vector<2xf32>, vector<3xf32>
//       CHECK:   %[[L0SH1:.*]] = vector.shuffle %[[D]], %[[E]] [0, 1, -1, 4] : vector<1xf32>, vector<5xf32>
//       CHECK:   %[[L0SH2:.*]] = vector.shuffle %[[B]], %[[B]] [0, -1, -1, -1] : vector<1xf32>, vector<1xf32>
//       CHECK:   %[[L1SH0:.*]] = vector.shuffle %[[L0SH0]], %[[L0SH1]] [0, 1, 4, 5, -1, 7] : vector<4xf32>, vector<4xf32>
//       CHECK:   %[[L2SH0:.*]] = vector.shuffle %[[L0SH2]], %[[L0SH2]] [0, -1, -1, -1, -1, -1] : vector<4xf32>, vector<4xf32>
//       CHECK:   %[[L3SH0:.*]] = vector.shuffle %[[L1SH0]], %[[L2SH0]] [0, 1, 2, 3, 6, 5] : vector<6xf32>, vector<6xf32>
//       CHECK:   return %[[L3SH0]] : vector<6xf32>

// -----

func.func @shuffle_tree_odd_intermediate_vectors(%a : vector<2xf32>,
                                                 %b : vector<2xf32>,
                                                 %c : vector<2xf32>,
                                                 %d : vector<2xf32>,
                                                 %e : vector<2xf32>,
                                                 %f : vector<2xf32>) -> vector<6xf32> {
  %0:2 = vector.to_elements %a : vector<2xf32>
  %1:2 = vector.to_elements %b : vector<2xf32>
  %2:2 = vector.to_elements %c : vector<2xf32>
  %3:2 = vector.to_elements %d : vector<2xf32>
  %4:2 = vector.to_elements %e : vector<2xf32>
  %5:2 = vector.to_elements %f : vector<2xf32>
  %6 = vector.from_elements %0#0, %1#1, %2#0, %3#1, %4#0, %5#1 : vector<6xf32>
  return %6 : vector<6xf32>
}

// CHECK-LABEL: func @shuffle_tree_odd_intermediate_vectors(
// CHECK-SAME:      %[[A:.*]]: vector<2xf32>, %[[B:.*]]: vector<2xf32>, %[[C:.*]]: vector<2xf32>, %[[D:.*]]: vector<2xf32>, %[[E:.*]]: vector<2xf32>, %[[F:.*]]: vector<2xf32>
//       CHECK:   %[[L0SH0:.*]] = vector.shuffle %[[A]], %[[B]] [0, 3] : vector<2xf32>, vector<2xf32>
//       CHECK:   %[[L0SH1:.*]] = vector.shuffle %[[C]], %[[D]] [0, 3] : vector<2xf32>, vector<2xf32>
//       CHECK:   %[[L0SH2:.*]] = vector.shuffle %[[E]], %[[F]] [0, 3] : vector<2xf32>, vector<2xf32>
//       CHECK:   %[[L1SH0:.*]] = vector.shuffle %[[L0SH0]], %[[L0SH1]] [0, 1, 2, 3] : vector<2xf32>, vector<2xf32>
//       CHECK:   %[[L2SH0:.*]] = vector.shuffle %[[L0SH2]], %[[L0SH2]] [0, 1, -1, -1] : vector<2xf32>, vector<2xf32>
//       CHECK:   %[[L3SH0:.*]] = vector.shuffle %[[L1SH0]], %[[L2SH0]] [0, 1, 2, 3, 4, 5] : vector<4xf32>, vector<4xf32>
//       CHECK:   return %[[L3SH0]] : vector<6xf32>
