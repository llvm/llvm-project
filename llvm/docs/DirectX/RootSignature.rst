===================================
Root Signature Serialization format
===================================

.. contents::
   :local:

Serialized format
=================

The root signature will be serialized into a binary format and saved in 'RTS0'
part of DXContainer.
The binary format is a sequence of bytes that can be used to create a root 
signature object in the Direct3D 12 API. The binary format is defined by the
`D3D12_ROOT_SIGNATURE_DESC (for rootsig_1_0)
<https://learn.microsoft.com/en-us/windows/win32/api/d3d12/ns-d3d12-d3d12_root_signature_desc>`_
or `D3D12_ROOT_SIGNATURE_DESC1 (for rootsig_1_1)
<https://learn.microsoft.com/en-us/windows/win32/api/d3d12/ns-d3d12-d3d12_root_signature_desc1>`_
structure in the Direct3D 12 API. (With the pointers translated to offsets.)

It will look like this:

.. code-block:: c++

  namespace dxbc {
    namespace SerializedRootSignature {
      namespace v_1_0 {

        struct DxilContainerDescriptorRange {
          uint32_t RangeType;
          uint32_t NumDescriptors;
          uint32_t BaseShaderRegister;
          uint32_t RegisterSpace;
          uint32_t OffsetInDescriptorsFromTableStart;
        };

        struct ContainerRootDescriptor {
          uint32_t ShaderRegister;
          uint32_t RegisterSpace;
        };
      }
      namespace v_1_1 {

        struct ContainerDescriptorRange {
          uint32_t RangeType;
          uint32_t NumDescriptors;
          uint32_t BaseShaderRegister;
          uint32_t RegisterSpace;
          uint32_t Flags;
          uint32_t OffsetInDescriptorsFromTableStart;
        };

        struct ContainerRootDescriptor {
          uint32_t ShaderRegister;
          uint32_t RegisterSpace;
          uint32_t Flags;
        };
      }

      struct ContainerRootDescriptorTable {
        uint32_t NumDescriptorRanges;
        uint32_t DescriptorRangesOffset;
      };

      struct RootConstants {
        uint32_t ShaderRegister;
        uint32_t RegisterSpace = 0;
        uint32_t Num32BitValues;
      };

      struct ContainerRootParameter {
        uint32_t ParameterType;
        uint32_t ShaderVisibility;
        uint32_t PayloadOffset;
      };

      struct StaticSamplerDesc {
        Filter Filter = Filter::ANISOTROPIC;
        TextureAddressMode AddressU = TextureAddressMode::Wrap;
        TextureAddressMode AddressV = TextureAddressMode::Wrap;
        TextureAddressMode AddressW = TextureAddressMode::Wrap;
        float MipLODBias = 0.f;
        uint32_t MaxAnisotropy = 16;
        ComparisonFunc ComparisonFunc = ComparisonFunc::LessEqual;
        StaticBorderColor BorderColor = StaticBorderColor::OpaqueWhite;
        float MinLOD = 0.f;
        float MaxLOD = MaxLOD;
        uint32_t ShaderRegister;
        uint32_t RegisterSpace = 0;
        ShaderVisibility ShaderVisibility = ShaderVisibility::All;
      };

      struct ContainerRootSignatureDesc {
        uint32_t Version;
        uint32_t NumParameters;
        uint32_t RootParametersOffset;
        uint32_t NumStaticSamplers;
        uint32_t StaticSamplersOffset;
        uint32_t Flags;
      };
    }
  }


The binary representation begins with a **ContainerRootSignatureDesc**
object.

The object is followed by an array of **ContainerRootParameter** objects
located at **ContainerRootSignatureDesc::RootParametersOffset**, which
corresponds to the size of **ContainerRootSignatureDesc**.

Next, there is an array of **StaticSamplerDesc** at
**ContainerRootSignatureDesc::StaticSamplersOffset**.

Following this, an array of **ContainerDescriptorRange** is presented,
which encompasses the descriptor ranges for all
**ContainerRootDescriptorTable** objects.

Subsequently, a detailed object (**RootConstants**,
**ContainerRootDescriptorTable**, or
**ContainerRootDescriptor**, depending on the parameter
type) for each **ContainerRootParameter** in the **ContainerRootParameter**
array. With **ContainerRootParameter.PayloadOffset** pointing to the 
detailed object.

In cases where the detailed object is a **ContainerRootDescriptorTable**,
**ContainerRootDescriptorTable.DescriptorRangesOffset** will point to an array
of **ContainerDescriptorRange**.

.. code-block::

    #define RS "RootFlags( ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT | " \
              "DENY_VERTEX_SHADER_ROOT_ACCESS), " \
              "CBV(b0, space = 1, flags = DATA_STATIC), " \
              "SRV(t0), " \
              "UAV(u0), " \
              "DescriptorTable( CBV(b1), " \
              "                 SRV(t1, numDescriptors = 8, " \
              "                     flags = DESCRIPTORS_VOLATILE), " \
              "                 UAV(u1, numDescriptors = unbounded, " \
              "                     flags = DESCRIPTORS_VOLATILE)), " \
              "DescriptorTable(Sampler(s0, space=1, numDescriptors = 4)), " \
              "RootConstants(num32BitConstants=3, b10), " \
              "StaticSampler(s1)," \
              "StaticSampler(s2, " \
              "              addressU = TEXTURE_ADDRESS_CLAMP, " \
              "              filter = FILTER_MIN_MAG_MIP_LINEAR )"

    [RootSignature(MyRS)]
    float4 main(float4 coord : COORD) : SV_Target
    {
    …
    }


The layout could be look like this for the MyRS in above example:

.. code-block:: c++

  struct SerializedRS {
    ContainerRootSignatureDesc RSDesc;
    ContainerRootParameter  rootParameters[6];
    StaticSamplerDesc  samplers[2];

    ContainerDescriptorRange ranges[3];

    // Extra part for each RootParameter in rootParameters.
    // RootConstants/RootDescriptorTable/RootDescriptor dependent on ParameterType.

    struct {

      ContainerRootDescriptor b0;

      ContainerRootDescriptor t1;

      ContainerRootDescriptor u1;

      ContainerRootDescriptorTable tab0; // tab0.DescriptorRangesOffset points to ranges[0]
                                    // tab0.NumDescriptorRanges = 2

      ContainerRootDescriptorTable tab1; // tab1.DescriptorRangesOffset points to ranges[2]
                                    // tab1.NumDescriptorRanges = 1

      RootConstants b10;

    };
    

  };

