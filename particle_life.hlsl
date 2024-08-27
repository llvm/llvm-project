#define ROOT_SIGNATURE \
    "RootFlags(ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT)," \
    "CBV(b0)," \
    "SRV(t0)," \
    "SRV(t1)," \
    "UAV(u0)," \
    "UAV(u1)"
    

cbuffer CONSTANTS : register(b0) {
    uint ParticleTypeMax;
    uint NumParticles;
    float2 WorldSize;
    float Friction;
    float ForceMultipler;
}

struct Rule {
    float force;
    float min_distance;
    float max_distance;
};

struct Particle {
    float2 position;
    float2 velocity;
    uint type;
};

struct Vertex {
    float2 position;
    uint color;
};

StructuredBuffer<Rule> Rules : register(t0);
StructuredBuffer<Particle> OldParticles : register(t1);
RWStructuredBuffer<Particle> NewParticles : register(u0);
RWStructuredBuffer<Vertex> Vertices : register(u1);


float3 particle_type_to_color(uint type);
uint float_to_abgr(float3 rgb);


[numthreads(32, 1, 1)]
void main(uint3 dispatch_thread_id : SV_DispatchThreadID) {
    uint particle_id = dispatch_thread_id.x;

    Particle particle = OldParticles[particle_id];
    
    // Accumulate forces
    float2 force = float2(0,0);
    float hit = 0;

    for (uint i = 0; i < NumParticles; ++i) {
        if (i == particle_id)
            continue;
        
        Particle other_particle = OldParticles[i];

        Rule rule = Rules[particle.type * ParticleTypeMax + other_particle.type];

        float2 direction = other_particle.position - particle.position;

        // wrapping
        if (direction.x > WorldSize.x * 0.5f)
            direction.x -= WorldSize.x;
        if (direction.x < WorldSize.x * -0.5f)
            direction.x += WorldSize.x;
        if (direction.y > WorldSize.y * 0.5f)
            direction.y -= WorldSize.y;
        if (direction.y < WorldSize.y * -0.5f)
            direction.y += WorldSize.y;

        // apply rule   
        float distance = length(direction);
        direction = normalize(direction);

        if (distance < rule.min_distance) {
            float repulsive_amount = abs(rule.force) * (1.0f - (distance / rule.min_distance))  * -3.0f;
            force += direction * repulsive_amount;
        }

        if (distance < rule.max_distance) {
            float attract_amount = rule.force * (1.0f - (distance / rule.max_distance));
            force += direction * attract_amount;
            hit += 0.01f;
        }
    }

    float2 velocity = particle.velocity;
    velocity += force * ForceMultipler;
    velocity *= Friction;

    particle.position = particle.position + velocity;

    if (particle.position.x < 0)
        particle.position.x += WorldSize.x;

    if (particle.position.x > WorldSize.x)
        particle.position.x -= WorldSize.x;

    if (particle.position.y < 0)
        particle.position.y += WorldSize.y;

    if (particle.position.y > WorldSize.y)
        particle.position.y -= WorldSize.y;


    particle.velocity = velocity;

    Vertices[particle_id].position = particle.position;

    float3 color =  particle_type_to_color(particle.type);

    color = lerp(color, color * 0.1f, 1-saturate(hit));

    Vertices[particle_id].color = float_to_abgr(color);

    NewParticles[particle_id] = particle;
}



// from https://chilliant.com/rgb2hsv.html
float3 hue2rgb(float H) {
    float R = abs(H * 6 - 3) - 1;  
    float G = 2 - abs(H * 6 - 2);  
    float B = 2 - abs(H * 6 - 4); 
    return saturate(float3(R,G,B));
}

float3 particle_type_to_color(uint type) {
    float hue = (float)type / float(ParticleTypeMax);
    return hue2rgb(hue);
}

uint float_to_abgr(float3 rgb) {
    rgb *= 255.0;

    uint r = rgb.x;
    uint g = rgb.y;
    uint b = rgb.z;
    uint a = 255; 

    return (a << 24) | (b << 16) | (g << 8) | r;    
}