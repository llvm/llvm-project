// RUN: mlir-opt %s -convert-vector-to-scf -lower-affine -convert-scf-to-cf -convert-vector-to-llvm="enable-amx" -finalize-memref-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts | \
// RUN: mlir-translate -mlir-to-llvmir | \
// RUN: %lli --entry-function=entry --mattr="+amx-tile,+amx-int8,+amx-bf16" --dlopen=%mlir_c_runner_utils | \
// RUN: FileCheck %s

// Note: To run this test, your CPU must support AMX.

func.func @print(%arg0: memref<16x4xi32>) {
  %iu = arith.constant -1: i32
  %c0 = arith.constant 0: index
  %c1 = arith.constant 1: index
  %c16 = arith.constant 16: index
  scf.for %i = %c0 to %c16 step %c1 {
    %0 = vector.transfer_read %arg0[%i, %c0], %iu: memref<16x4xi32>, vector<4xi32>
    vector.print %0 : vector<4xi32>
  }
  return
}

func.func @kernel1(%arg0: memref<16x16xi8>,
              %arg1: memref<4x16xi8>,
              %arg2: memref<16x4xi32>) {
  %0 = arith.constant 0 : index
  %1 = amx.tile_load %arg0[%0, %0] : memref<16x16xi8>  into vector<16x16xi8>
  %2 = amx.tile_load %arg1[%0, %0] : memref<4x16xi8>  into vector<4x16xi8>
  %3 = amx.tile_zero : vector<16x4xi32>
  %4 = amx.tile_muli %1, %2, %3 : vector<16x16xi8>, vector<4x16xi8>, vector<16x4xi32>
  amx.tile_store %arg2[%0, %0], %4 : memref<16x4xi32>, vector<16x4xi32>
  return
}

func.func @kernel2(%arg0: memref<16x16xi8>,
              %arg1: memref<4x16xi8>,
              %arg2: memref<16x4xi32>) {
  %0 = arith.constant 0 : index
  %1 = amx.tile_load %arg0[%0, %0] : memref<16x16xi8>  into vector<16x16xi8>
  %2 = amx.tile_load %arg1[%0, %0] : memref<4x16xi8>  into vector<4x16xi8>
  %3 = amx.tile_zero : vector<16x4xi32>
  %4 = amx.tile_muli %1, %2 zext, %3 : vector<16x16xi8>, vector<4x16xi8>, vector<16x4xi32>
  amx.tile_store %arg2[%0, %0], %4 : memref<16x4xi32>, vector<16x4xi32>
  return
}

func.func @kernel3(%arg0: memref<16x16xi8>,
              %arg1: memref<4x16xi8>,
              %arg2: memref<16x4xi32>) {
  %0 = arith.constant 0 : index
  %1 = amx.tile_load %arg0[%0, %0] : memref<16x16xi8>  into vector<16x16xi8>
  %2 = amx.tile_load %arg1[%0, %0] : memref<4x16xi8>  into vector<4x16xi8>
  %3 = amx.tile_zero : vector<16x4xi32>
  %4 = amx.tile_muli %1 zext, %2, %3 : vector<16x16xi8>, vector<4x16xi8>, vector<16x4xi32>
  amx.tile_store %arg2[%0, %0], %4 : memref<16x4xi32>, vector<16x4xi32>
  return
}

func.func @kernel4(%arg0: memref<16x16xi8>,
              %arg1: memref<4x16xi8>,
              %arg2: memref<16x4xi32>) {
  %0 = arith.constant 0 : index
  %1 = amx.tile_load %arg0[%0, %0] : memref<16x16xi8>  into vector<16x16xi8>
  %2 = amx.tile_load %arg1[%0, %0] : memref<4x16xi8>  into vector<4x16xi8>
  %3 = amx.tile_zero : vector<16x4xi32>
  %4 = amx.tile_muli %1 zext, %2 zext, %3 : vector<16x16xi8>, vector<4x16xi8>, vector<16x4xi32>
  amx.tile_store %arg2[%0, %0], %4 : memref<16x4xi32>, vector<16x4xi32>
  return
}

func.func @entry() -> i32 {
  %c0 = arith.constant 0: index

  // Set up memory.
  %a = memref.alloc() : memref<16x16xi8>
  %b = memref.alloc() : memref<4x16xi8>
  %c = memref.alloc() : memref<16x4xi32>

  %0 = arith.constant dense<
    [ [  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,  14,  15 ],
      [ 16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,  28,  29,  30,  31 ],
      [ 32,  33,  34,  35,  36,  37,  38,  39,  40,  41,  42,  43,  44,  45,  46,  47 ],
      [ 48,  49,  50,  51,  52,  53,  54,  55,  56,  57,  58,  59,  60,  61,  62,  63 ],
      [ 64,  65,  66,  67,  68,  69,  70,  71,  72,  73,  74,  75,  76,  77,  78,  79 ],
      [ 80,  81,  82,  83,  84,  85,  86,  87,  88,  89,  90,  91,  92,  93,  94,  95 ],
      [ 96,  97,  98,  99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111 ],
      [112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127 ],
      [128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143 ],
      [144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159 ],
      [160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175 ],
      [176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191 ],
      [192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207 ],
      [208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223 ],
      [224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239 ],
      [240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255 ] ]> : vector<16x16xi8>

  %1 = arith.constant dense<
    [ [192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207 ],
      [208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223 ],
      [224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239 ],
      [240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255 ] ]> : vector<4x16xi8>

  vector.transfer_write %0, %a[%c0, %c0] : vector<16x16xi8>, memref<16x16xi8>
  vector.transfer_write %1, %b[%c0, %c0] : vector<4x16xi8>,  memref<4x16xi8>

  // Call kernel1 and verify result.
  //
  // CHECK:      ( -3320, -2840, -2360, -1880 )
  // CHECK-NEXT: ( -13176, -11672, -10168, -8664 )
  // CHECK-NEXT: ( -23032, -20504, -17976, -15448 )
  // CHECK-NEXT: ( -32888, -29336, -25784, -22232 )
  // CHECK-NEXT: ( -42744, -38168, -33592, -29016 )
  // CHECK-NEXT: ( -52600, -47000, -41400, -35800 )
  // CHECK-NEXT: ( -62456, -55832, -49208, -42584 )
  // CHECK-NEXT: ( -72312, -64664, -57016, -49368 )
  // CHECK-NEXT: ( 75528, 67816, 60104, 52392 )
  // CHECK-NEXT: ( 65672, 58984, 52296, 45608 )
  // CHECK-NEXT: ( 55816, 50152, 44488, 38824 )
  // CHECK-NEXT: ( 45960, 41320, 36680, 32040 )
  // CHECK-NEXT: ( 36104, 32488, 28872, 25256 )
  // CHECK-NEXT: ( 26248, 23656, 21064, 18472 )
  // CHECK-NEXT: ( 16392, 14824, 13256, 11688 )
  // CHECK-NEXT: ( 6536, 5992, 5448, 4904 )
  //
  call @kernel1(%a, %b, %c) : (memref<16x16xi8>, memref<4x16xi8>, memref<16x4xi32>) -> ()
  call @print(%c) : (memref<16x4xi32>) -> ()

  // Call kernel2 and verify result.
  //
  // CHECK-NEXT: ( 27400, 27880, 28360, 28840 )
  // CHECK-NEXT: ( 83080, 84584, 86088, 87592 )
  // CHECK-NEXT: ( 138760, 141288, 143816, 146344 )
  // CHECK-NEXT: ( 194440, 197992, 201544, 205096 )
  // CHECK-NEXT: ( 250120, 254696, 259272, 263848 )
  // CHECK-NEXT: ( 305800, 311400, 317000, 322600 )
  // CHECK-NEXT: ( 361480, 368104, 374728, 381352 )
  // CHECK-NEXT: ( 417160, 424808, 432456, 440104 )
  // CHECK-NEXT: ( -418040, -425752, -433464, -441176 )
  // CHECK-NEXT: ( -362360, -369048, -375736, -382424 )
  // CHECK-NEXT: ( -306680, -312344, -318008, -323672 )
  // CHECK-NEXT: ( -251000, -255640, -260280, -264920 )
  // CHECK-NEXT: ( -195320, -198936, -202552, -206168 )
  // CHECK-NEXT: ( -139640, -142232, -144824, -147416 )
  // CHECK-NEXT: ( -83960, -85528, -87096, -88664 )
  // CHECK-NEXT: ( -28280, -28824, -29368, -29912 )
  //
  call @kernel2(%a, %b, %c) : (memref<16x16xi8>, memref<4x16xi8>, memref<16x4xi32>) -> ()
  call @print(%c) : (memref<16x4xi32>) -> ()

  // Call kernel3 and verify result.
  //
  // CHECK-NEXT: ( -3320, -2840, -2360, -1880 )
  // CHECK-NEXT: ( -13176, -11672, -10168, -8664 )
  // CHECK-NEXT: ( -23032, -20504, -17976, -15448 )
  // CHECK-NEXT: ( -32888, -29336, -25784, -22232 )
  // CHECK-NEXT: ( -42744, -38168, -33592, -29016 )
  // CHECK-NEXT: ( -52600, -47000, -41400, -35800 )
  // CHECK-NEXT: ( -62456, -55832, -49208, -42584 )
  // CHECK-NEXT: ( -72312, -64664, -57016, -49368 )
  // CHECK-NEXT: ( -82168, -73496, -64824, -56152 )
  // CHECK-NEXT: ( -92024, -82328, -72632, -62936 )
  // CHECK-NEXT: ( -101880, -91160, -80440, -69720 )
  // CHECK-NEXT: ( -111736, -99992, -88248, -76504 )
  // CHECK-NEXT: ( -121592, -108824, -96056, -83288 )
  // CHECK-NEXT: ( -131448, -117656, -103864, -90072 )
  // CHECK-NEXT: ( -141304, -126488, -111672, -96856 )
  // CHECK-NEXT: ( -151160, -135320, -119480, -103640 )
  //
  call @kernel3(%a, %b, %c) : (memref<16x16xi8>, memref<4x16xi8>, memref<16x4xi32>) -> ()
  call @print(%c) : (memref<16x4xi32>) -> ()

  // Call kernel4 and verify result.
  //
  // CHECK-NEXT: ( 27400, 27880, 28360, 28840 )
  // CHECK-NEXT: ( 83080, 84584, 86088, 87592 )
  // CHECK-NEXT: ( 138760, 141288, 143816, 146344 )
  // CHECK-NEXT: ( 194440, 197992, 201544, 205096 )
  // CHECK-NEXT: ( 250120, 254696, 259272, 263848 )
  // CHECK-NEXT: ( 305800, 311400, 317000, 322600 )
  // CHECK-NEXT: ( 361480, 368104, 374728, 381352 )
  // CHECK-NEXT: ( 417160, 424808, 432456, 440104 )
  // CHECK-NEXT: ( 472840, 481512, 490184, 498856 )
  // CHECK-NEXT: ( 528520, 538216, 547912, 557608 )
  // CHECK-NEXT: ( 584200, 594920, 605640, 616360 )
  // CHECK-NEXT: ( 639880, 651624, 663368, 675112 )
  // CHECK-NEXT: ( 695560, 708328, 721096, 733864 )
  // CHECK-NEXT: ( 751240, 765032, 778824, 792616 )
  // CHECK-NEXT: ( 806920, 821736, 836552, 851368 )
  // CHECK-NEXT: ( 862600, 878440, 894280, 910120 )
  //
  call @kernel4(%a, %b, %c) : (memref<16x16xi8>, memref<4x16xi8>, memref<16x4xi32>) -> ()
  call @print(%c) : (memref<16x4xi32>) -> ()

  // Release resources.
  memref.dealloc %a : memref<16x16xi8>
  memref.dealloc %b : memref<4x16xi8>
  memref.dealloc %c : memref<16x4xi32>

  %i0 = arith.constant 0 : i32
  return %i0 : i32
}
